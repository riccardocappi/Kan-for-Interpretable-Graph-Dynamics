from tsl.datasets import MetrLA, PemsBay
from .SpatioTemporalGraph import SpatioTemporalGraph
import torch
import os
from tsl.data.preprocessing.scalers import MinMaxScaler

traffic_data_name = ['metrla', 'pemsbay']

class TrafficData(SpatioTemporalGraph):
    def __init__(
        self, 
        root,
        name,
        num_samples,
        seed,
        device='cpu',
        n_ics = 3
    ):       
        
        assert name in traffic_data_name        
        super().__init__(
            root=root,
            name=name,
            n_ics=n_ics,
            n_samples=num_samples,
            seed=seed,
            device=device
        )
        

    def get_raw_data(self):
        if self.name == 'metrla':
            dataset = MetrLA(os.path.join(self.root, self.name), impute_zeros=True)
        elif self.name == 'pemsbay':
            dataset = PemsBay(self.root)
        else:
            raise NotImplementedError()
        
        edge_index, edge_attr = dataset.get_connectivity(
            threshold=0.1,
            include_self=False,
            normalize_axis=1,
            layout="edge_index"
        )
        
        df = dataset.dataframe()
        
        scaler = MinMaxScaler(out_range=(-1, 1))
        scaled_data = scaler.fit_transform(df.values)
        
        raw_data = torch.from_numpy(scaled_data).unsqueeze(2)
                
        # Reshaping tensor as (ICs, num_samples, num_modes, 1), where each initial condition is a different day
        time_steps, n_nodes, _ = raw_data.shape
        sampling_frequency = 288 # one measurement every 5 minutes → 288 samples/day
        tot_days = time_steps // sampling_frequency
        raw_data = raw_data.view(tot_days, sampling_frequency, n_nodes, 1)
        raw_data = raw_data[:self.n_ics]    # Consider the first n_ics days
        
        raw_data = raw_data.to(torch.device(self.device))
        time = torch.linspace(0, 1, raw_data.size(1)).repeat(raw_data.size(0), 1).to(torch.device(self.device))
        edge_index = torch.from_numpy(edge_index).to(torch.device(self.device))
        edge_attr = torch.from_numpy(edge_attr).to(torch.device(self.device))
        
        return edge_index, edge_attr, raw_data, time 
        