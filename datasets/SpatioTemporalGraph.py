from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from utils.utils import sample_irregularly_per_ics

class SpatioTemporalGraph(InMemoryDataset, ABC):
    def __init__(
        self, 
        root,
        name,
        n_samples,
        seed,
        device='cpu',
        history = 1,
        horizon = 1,
        n_ics = 3,
        stride=24,
        noise_scale=0.0
    ):
        self.name = name
        self.num_samples = n_samples
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.device = device
        assert horizon >= 3
        self.horizon = horizon
        self.history = history
        self.n_ics = n_ics
        self.stride = stride
        self.nosie_scale = noise_scale
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
        
        if self.num_samples > 0:
            raw_data, time = sample_irregularly_per_ics(raw_data, time, self.num_samples)
        
        input_length = self.history
        target_length = self.horizon
        total_seq_len = input_length + target_length
                
        data = []
        
        for ic in range(raw_data.size(0)):
            
            mean_value = raw_data[ic].mean().item()
            noise_strength = self.nosie_scale * mean_value
            nosie = self.rng.normal(0, noise_strength, size=raw_data[ic].shape)
            noise = torch.from_numpy(nosie).float().to(self.device)
            raw_data[ic] += noise
            
            for ts in range(0, raw_data.size(1) - total_seq_len + 1, self.stride):
                idx_input = slice(ts, ts + input_length)
                idx_target = slice(ts + input_length, ts + total_seq_len)
                
                x = raw_data[ic, idx_input, :, :]  # Shape: (input_length, num_nodes, 1)
                y = raw_data[ic, idx_target, :, :]  # Shape: (target_length, num_nodes, 1)
                   
                t_span = time[ic, ts + input_length-1: ts + total_seq_len]
                
                backprop_idx = torch.tensor([0, self.horizon//2, -1], device=self.device)
                
                data.append(
                    Data(
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        x=x,  
                        y=y,
                        t_span=t_span,
                        backprop_idx=backprop_idx
                    )
                )
        
        data, slices = self.collate(data)
        torch.save((data, slices, raw_data, time), self.processed_paths[0])
        
    
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError()