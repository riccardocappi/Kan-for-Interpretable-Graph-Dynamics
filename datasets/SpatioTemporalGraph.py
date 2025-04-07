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
        n_ics,
        n_samples,
        seed,
        device='cpu'
    ):
        self.name = name
        self.num_samples = n_samples
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.device = device
        self.n_ics = n_ics
        super().__init__(root)
        self.data, self.slices, self.raw_data_sampled = torch.load(self.processed_paths[0])
        
        
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
        
        data_sampled, t_sampled = sample_irregularly_per_ics(raw_data, time, self.num_samples)
        
        data = []
        ics, n_samples, _, _ = data_sampled.shape
        len_data = ics * (n_samples - 1)
        
        for i in range(len_data):
            index_ic = i // (data_sampled.size(1) - 1)
            index_sample = i % (data_sampled.size(1) - 1)
            
            x_in = data_sampled[index_ic, index_sample, : ,:]
            y_target = data_sampled[index_ic, index_sample + 1, :, :]
            t_start = t_sampled[index_ic, index_sample]
            t_target = t_sampled[index_ic, index_sample + 1]
            
            data.append(
                Data(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    x = x_in,
                    y = y_target,
                    t_start = t_start,
                    t_target = t_target
                )
            )
            
        data, slices = self.collate(data)
        torch.save((data, slices, data_sampled), self.processed_paths[0]) 
        
    
    @abstractmethod
    def get_raw_data(self):
        raise NotImplementedError()