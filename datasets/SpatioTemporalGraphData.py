from utils.utils import integrate, sample_irregularly_per_ics
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import torch
import os
import networkx as nx
from torch_geometric.utils import from_networkx


class SpatioTemporalGraphData(InMemoryDataset):
    def __init__(
        self, 
        root,
        dynamics,
        t_span = [0, 1],
        t_max = 300,
        num_samples = 30,
        seed=42, 
        n_ics = 3,
        input_range = [0,1],
        device='cpu',
        **integration_kwargs
    ):  
        self.rng = np.random.default_rng(seed=seed)
        self.dynamics = dynamics
        self.t_span = t_span
        self.t_max = t_max
        self.num_samples = num_samples
        self.seed = seed
        self.n_ics = n_ics
        self.input_range = input_range
        self.integration_kwargs = integration_kwargs
        self.device = device
        
        super().__init__(root=root)
        self.data, self.slices, self.raw_data_sampled = torch.load(self.processed_paths[0])
        
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.dynamics, 'processed')
    
    
    @property
    def processed_file_names(self):
        name = f'{self.dynamics}.pt'
        return [name]
    
    
    def process(self):
        print('Builing the dataset...')
        
        graph = nx.barabasi_albert_graph(70, 3, seed=self.seed)
        edge_index = from_networkx(graph).edge_index
        edge_index = edge_index.to(torch.device(self.device))
        
        
        raw_data, t = [], []
        for _ in range(self.n_ics):
            data_k, t_k = integrate(
                input_range=self.input_range,
                t_span=self.t_span,
                t_eval_steps=self.t_max,
                dynamics=self.dynamics,
                device=torch.device(self.device),
                graph=graph,
                rng=self.rng,
                **self.integration_kwargs
            )
            raw_data.append(data_k)
            t.append(t_k)
        
        raw_data = torch.stack(raw_data, dim=0) #(n_ics, t_max, n_nodes, n_features)
        t = torch.stack(t, dim=0)               #(n_ics, t_max)
                
        data_sampled, t_sampled = sample_irregularly_per_ics(raw_data, t, self.num_samples)
        data = []
        
        len_data = self.n_ics * (self.num_samples - 1)
        
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
                    x = x_in,
                    y = y_target,
                    t_start = t_start,
                    t_target = t_target
                )
            )

        data, slices = self.collate(data)
        torch.save((data, slices, data_sampled), self.processed_paths[0])  