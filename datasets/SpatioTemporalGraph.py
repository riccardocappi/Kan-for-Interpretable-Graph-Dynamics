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
        horizon = 1,
        n_ics = 3
    ):
        self.name = name
        self.num_samples = n_samples
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.device = device
        assert horizon > 0
        self.horizon = horizon
        self.n_ics = n_ics
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
        print('Builing the dataset...')
        
        edge_index, edge_attr, raw_data, time = self.get_raw_data()
        assert (raw_data.size(0) == time.size(0)) and (raw_data.size(1) == time.size(1))
        
        data_sampled, t_sampled, indices = sample_irregularly_per_ics(raw_data, time, self.num_samples)
        
        data = []
        
        for ic in range(indices.size(0)):
            for i, ts in enumerate(indices[ic, :-self.horizon]):
                x = raw_data[ic, ts, :, :]
                idx =  indices[ic, i:i + self.horizon + 1]
                if len(idx) > 1:
                    # Select 10% of idx, but always include the first index (i.e., x's timestamp)
                    n_select = max(2, int(len(idx) * 0.1))  # at least 2 to ensure x and y exist
                    selected_indices = torch.randperm(len(idx), device=idx.device)[:n_select]
                    selected_indices = torch.cat([torch.tensor([0], device=idx.device), selected_indices])
                    selected_indices = torch.unique(selected_indices)  # remove duplicates if 0 already selected
                    selected_indices = selected_indices[torch.argsort(selected_indices)]  # keep order
                    idx = idx[selected_indices]
                
                t_span = time[ic, idx]
                y = raw_data[ic, idx[1:]]
                x_mask = (x != 0).unsqueeze(0)
                y_mask = (y != 0)
                mask = x_mask & y_mask
                
                data.append(
                    Data(
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        x = x,
                        y = y,
                        t_span = t_span,
                        mask = mask
                    )
                )
                
        data, slices = self.collate(data)
        torch.save((data, slices, data_sampled, t_sampled), self.processed_paths[0]) 
        
    
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError()