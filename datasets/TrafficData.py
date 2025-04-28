from tsl.datasets import MetrLA, PemsBay
from .SpatioTemporalGraph import SpatioTemporalGraph
import torch
import os

traffic_data_name = ['metrla', 'pemsbay', 'metrla2']

class TrafficData(SpatioTemporalGraph):
    def __init__(
        self, 
        root,
        name,
        num_samples,
        seed,
        device='cpu',
        n_ics = 3,
        horizon = 1
    ):       
        
        assert name in traffic_data_name        
        super().__init__(
            root=root,
            name=name,
            n_ics=n_ics,
            n_samples=num_samples,
            seed=seed,
            device=device,
            horizon=horizon
        )
        

    def get_raw_data(self):
        if self.name == 'metrla' or self.name == 'metrla2':
            dataset = MetrLA(os.path.join(self.root, self.name), impute_zeros=False)
        else:
            raise NotImplementedError()
        
        edge_index, edge_attr = dataset.get_connectivity(
            threshold=0.1,
            include_self=False,
            normalize_axis=1,
            layout="edge_index"
        )
        
        df = dataset.dataframe()        
        raw_data = torch.from_numpy(df.values).unsqueeze(2)
        
        time_steps, n_nodes, _ = raw_data.shape
        samples_per_day = 288
        samples_per_week = 7 * samples_per_day
        tot_weeks = time_steps // samples_per_week
        raw_data = raw_data[:tot_weeks * samples_per_week]  # Trim to full weeks
        raw_data = raw_data.view(tot_weeks, samples_per_week, n_nodes, 1)
        to_keep = self.n_ics if self.n_ics > 0 else tot_weeks
        raw_data = raw_data[:to_keep]  # First n_ics weeks

        raw_data = raw_data.to(torch.device(self.device))
        time = torch.linspace(0, 1, raw_data.size(1)).repeat(raw_data.size(0), 1).to(torch.device(self.device))
        edge_index = torch.from_numpy(edge_index).to(torch.device(self.device))
        edge_attr = torch.from_numpy(edge_attr).to(torch.device(self.device))
        
        return edge_index, edge_attr, raw_data, time