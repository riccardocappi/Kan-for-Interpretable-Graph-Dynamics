from torch_geometric.data import Data, InMemoryDataset
from networkx import Graph
from typing import Callable, Optional
import numpy as np
from numpy.random import Generator, default_rng
import torch
import os
from .data_utils import numerical_integration, sample_with_minimum_distance
from torch_geometric.utils import from_networkx
import tqdm


class GraphDynamics(InMemoryDataset):
    def __init__(self, 
                root: str,
                name: str,
                dynamics: str,
                graph: Graph,
                time_steps: int = 1000,
                num_samples: int = 1000,
                step_size = 0.01,
                min_sample_distance: int = 1,
                name_suffix: str = '',
                rng: Optional[Generator] = None,
                regular_samples:bool = True,
                add_noise_target:bool = False,
                noise_strength:float = 0.01,
                add_noise_input:bool = False,
                in_noise_dim: int = 1,
                input_range=None,
                n_iters=1,
                **kwargs # Additional arguments passed to numerical_integration function
                ):
                
        self.root = root
        self.name = name
        self.suffix = name_suffix
        
        self.dynamics = dynamics
        self.G = graph
        self.time_steps = time_steps
        self.num_samples = num_samples
        self.input_range = input_range
        self.min_sample_distance = min_sample_distance
        self.rng = rng if rng is not None else default_rng()

        self.step_size = step_size
        self.regular_samples = regular_samples
        self.add_noise_target = add_noise_target
        self.noise_strength = noise_strength
        self.add_noise_input = add_noise_input
        self.in_noise_dim = in_noise_dim
        self.n_iters = n_iters
        self.kwargs = kwargs
        
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.input_dim = self[0].x.shape[-1]
        self.output_dim = self[0].y.shape[-1]
        self.time_dim = None
        

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        name = f'{self.name}_{self.suffix}.pt'
        return [name]
    
    
    def process(self):
        N = self.G.number_of_nodes()
        data = []
        for _ in range(self.n_iters): 
        
            initial_state = self.rng.uniform(0,1,N) if self.input_range is None else self.rng.uniform(self.input_range[0],
                                                                                                    self.input_range[1],
                                                                                                    N)
            xs = numerical_integration(self.G, self.dynamics, initial_state, self.time_steps, self.step_size, **self.kwargs)
            xs = np.expand_dims(xs, 2)
            
            if self.add_noise_input:
                # Input noise
                random_features = self.rng.uniform(0, 1, (xs.shape[0], xs.shape[1], self.in_noise_dim))
                xs = np.concatenate([xs, random_features], axis=-1)
                
                
            edge_index = from_networkx(self.G).edge_index
            
            if not self.regular_samples:
                to_keep = sample_with_minimum_distance(
                    n = self.time_steps, 
                    k = self.num_samples, 
                    d = self.min_sample_distance, 
                    rng = self.rng
                )
            else:
                to_keep = range(1, len(xs))
            
            prev_x = torch.from_numpy(xs[0]).float()
            prev_i = 0
            for i in tqdm.tqdm(to_keep):
                y = torch.from_numpy(xs[i]).float()
                if self.add_noise_target:
                    # target noise
                    noise_y = y + self.rng.normal(0, self.noise_strength, y.shape)
                target = y if not self.add_noise_target else noise_y
                data.append(
                    Data(
                        edge_index=edge_index,
                        x = prev_x,
                        y = target,
                        delta_t = i - prev_i
                    )
                )
                prev_x = y
                prev_i = i
        
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])