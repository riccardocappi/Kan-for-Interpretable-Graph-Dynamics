from .SpatioTemporalGraph import SpatioTemporalGraph
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from utils.utils import integrate

class SyntheticData(SpatioTemporalGraph):
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
        horizon = 15,
        history= 1,
        stride=24,
        noise_scale=0.0,
        **integration_kwargs
    ):  
        
        self.t_span = t_span
        self.t_max = t_max
        self.input_range = input_range
        self.int_kwargs = integration_kwargs
        
        super().__init__(
            root=root,
            name=dynamics,
            n_ics=n_ics,
            n_samples=num_samples,
            seed=seed,
            device=device,
            horizon=horizon,
            history=history,
            stride=stride,
            noise_scale=noise_scale
        )
    
    
    def get_raw_data(self):
        
        graph = nx.barabasi_albert_graph(70, 3, seed=self.seed)
        edge_index = from_networkx(graph).edge_index
        edge_index = edge_index.to(torch.device(self.device))
        
        raw_data, t = [], []
        for _ in range(self.n_ics):
            data_k, t_k = integrate(
                input_range=self.input_range,
                t_span=self.t_span,
                t_eval_steps=self.t_max,
                dynamics=self.name,
                device=torch.device(self.device),
                graph=graph,
                rng=self.rng,
                **self.int_kwargs
            )
            raw_data.append(data_k)
            t.append(t_k)
        
        raw_data = torch.stack(raw_data, dim=0) #(n_ics, t_max, n_nodes, n_features)
        t = torch.stack(t, dim=0)               #(n_ics, t_max)
        
        return edge_index, None, raw_data, t
        