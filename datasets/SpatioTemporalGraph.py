from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from utils.utils import sample_irregularly_per_ics


def interp_points(raw_data, degree=3):
    raw_data_smoothed = []
    
    for ic in range(raw_data.size(0)):
        signal = raw_data[ic].detach().cpu().numpy()
        T,N, _ = signal.shape
        signal_smoothed = np.zeros_like(signal)
        x = np.arange(T)
        for node_idx in range(N):
            y = signal[:, node_idx, 0]
            coeffs = np.polyfit(x, y, degree)
            y_smooth = np.polyval(coeffs, x)
            signal_smoothed[:, node_idx, 0] = y_smooth
        raw_data_smoothed.append(torch.tensor(signal_smoothed, dtype=raw_data.dtype, device=raw_data.device))
    
    return torch.stack(raw_data_smoothed, dim=0)


class SpatioTemporalGraph(InMemoryDataset, ABC):
    def __init__(
        self, 
        root,
        name,
        n_samples,
        seed,
        device='cpu',
        history = 1,
        horizon = 15,
        stride=24,
        predict_deriv=False,
        denoise=False
    ):
        self.name = name
        self.num_samples = n_samples
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.device = device
        self.horizon = horizon if not predict_deriv else 1
        self.history = history if not predict_deriv else 1
        self.denoise = denoise
        self.stride = stride
        self.predict_deriv = predict_deriv
        super().__init__(root)
        self.data, self.slices, self.raw_data_sampled, self.t_sampled = torch.load(self.processed_paths[0])
        
        
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')
    
    
    @property
    def processed_file_names(self):
        name = f'{self.name}.pt'
        return [name]


    def process(self):
        print('Building the dataset...')

        edge_index, edge_attr, raw_data, time = self.get_raw_data()
        assert (raw_data.size(0) == time.size(0)) and (raw_data.size(1) == time.size(1))
        
        if self.denoise:
            raw_data = interp_points(raw_data, degree=3)
        
        if self.num_samples > 0:
            raw_data, time = sample_irregularly_per_ics(raw_data, time, self.num_samples)
        
        input_length = self.history
        target_length = self.horizon
        total_seq_len = input_length + target_length
        
        data = []
        
        for ic in range(raw_data.size(0)):
            first_derivatives = self.compute_five_point_fd(raw_data[ic], time)
            
            for ts in range(0, raw_data.size(1) - total_seq_len + 1, self.stride):
                idx_input = slice(ts, ts + input_length)
                idx_target = slice(ts + input_length, ts + total_seq_len)
                
                x = raw_data[ic, idx_input, :, :]  # Shape: (input_length, num_nodes, 1)
                
                if self.predict_deriv:
                    x = x.squeeze(0)
                    y = first_derivatives[ts, :, :]
                    backprop_idx = torch.tensor([], device=self.device)
                    t_span = torch.tensor([], device=self.device)
                else: 
                    y = raw_data[ic, idx_target, :, :]  # Shape: (target_length, num_nodes, 1)
                    backprop_idx = torch.tensor([0, self.horizon//2, -1], device=self.device)
                    t_span = time[ic, ts + input_length-1: ts + total_seq_len]
                
                data.append(
                    Data(
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        x=x,  
                        y=y,
                        t_span=t_span,
                        backprop_idx=backprop_idx,
                    )
                )
        
        data, slices = self.collate(data)
        torch.save((data, slices, raw_data, time), self.processed_paths[0])
        
    
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError()
    
    
    def compute_five_point_fd(self, raw_data, time):
        delta_t = time[0, 1] - time[0, 0]
        delta_t = delta_t.item()

        T, _, _ = raw_data.shape
        derivative = torch.zeros_like(raw_data)

        # Apply the five-point stencil to the interior points
        for t in range(2, T - 2):
            derivative[t] = (
                -raw_data[t + 2] + 8 * raw_data[t + 1] - 8 * raw_data[t - 1] + raw_data[t - 2]
            ) / (12 * delta_t)

        # Handle boundary values with lower-order differences (e.g., forward/backward)
        derivative[0] = (raw_data[1] - raw_data[0]) / delta_t
        derivative[1] = (raw_data[2] - raw_data[0]) / (2 * delta_t)
        derivative[-2] = (raw_data[-1] - raw_data[-3]) / (2 * delta_t)
        derivative[-1] = (raw_data[-1] - raw_data[-2]) / delta_t

        return derivative