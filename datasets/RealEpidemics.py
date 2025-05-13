from .SpatioTemporalGraph import SpatioTemporalGraph
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import torch
import json
import os


class RealEpidemics(SpatioTemporalGraph):
    def __init__(
        self, 
        root, 
        name, 
        n_samples=-1, 
        seed=42, 
        device='cpu', 
        history=1, 
        horizon=1, 
        n_ics=-1, 
        stride=1, 
        noise_scale=0.0
    ):
        super().__init__(
            root, 
            name, 
            n_samples=n_samples, 
            seed=seed, 
            device=device, 
            history=history, 
            horizon=horizon, 
            n_ics=n_ics, 
            stride=stride, 
            noise_scale = noise_scale
        )
    
    
    def get_raw_data(self):
        x, A, populations, countries = self.load_data()
        # Extract x_values (avoid time column)
        x_values = x[:, 1:]
        
        no_infected = np.where(np.mean(x_values, axis=0) == 0)[0]
        # Remove those columns/entries from all relevant data
        populations = np.delete(populations, no_infected, axis=0)
        x_values = np.delete(x_values, no_infected, axis=1)
        A = np.delete(A, no_infected, axis=0)
        A = np.delete(A, no_infected, axis=1)
        countries = np.delete(countries, no_infected, axis=0)
        
        x, x_values = self.inter_points(x, x_values)
        
        # Remove self-loops in A (zero the diagonal)
        Aij = A - np.diag(np.diag(A))
        PSI = 8.91e6
        Aij_act = Aij * PSI / np.sum(populations)
        
        infected = []
        for i in range(x_values.shape[1]):
            start_time = np.argmax(x_values[:, i] > 0)  # First index where x_values > 0
            infected.append(x_values[start_time:, i])

        data = []
        Ind = []
        j = 0
        Period = 45

        for i, inf in enumerate(infected):
            if len(inf) >= Period and inf[Period - 1] >= 100:
                data.append(inf[:Period])
                Ind.append(i)
                j += 1
                
        data = np.column_stack(data)
        
        all_indices = np.arange(len(Aij_act))
        cut_off = np.setdiff1d(all_indices, Ind)
        
        mask = np.ones(len(A), dtype=bool)
        mask[cut_off] = False

        populations = populations[mask]
        countries = countries[mask]
        A = A[mask][:, mask]
        Aij_act = Aij_act[mask][:, mask]
        
        countries_dict = dict(zip(countries, range(len(countries))))
        with open(os.path.join(self.root, self.name, "countries_dict.json"), 'w') as f:
            json.dump(countries_dict, f)
        
        data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(-1).to(self.device)
        
        row, col = np.nonzero(Aij_act)
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long, device=self.device)
        
        weights = Aij_act[row, col]  # edge weights corresponding to (row, col)
        edge_attr = torch.tensor(weights, dtype=torch.float, device=self.device).unsqueeze(-1)  # Shape: (num_edges, 1)
        
        time = torch.linspace(0, 1, data.size(1)).repeat(data.size(0), 1).to(torch.device(self.device))
        
        return edge_index, edge_attr, data, time

    
    def load_data(self):
        # Read infection data
        x = np.loadtxt('./data/RealEpidemics/infected_numbers_H1N1.csv', delimiter=',')

        # Read adjacency matrix (flight data)
        A = np.loadtxt('./data/RealEpidemics/Flights_adj.csv', delimiter=',')

        # Read populations
        populations = np.loadtxt('./data/RealEpidemics/populations.csv', delimiter=',')

        # Read country names (assuming second column contains names)
        countries_df = pd.read_csv('./data/RealEpidemics/Country_Population_final.csv')
        countries = countries_df.iloc[:, 1].astype(str).values  # Convert to string array

        return x, A, populations, countries
    
    
    def inter_points(self, x, x_values):
        x = np.column_stack((x[:, 0], x_values))

        # Extract timepoints
        timepoints = x[:, 0].astype(int)
        fully_timepoints = np.arange(0, int(x[-1, 0]) + 1)

        # Find missing timepoints
        missing_timepoints = np.setdiff1d(fully_timepoints, timepoints)

        # Interpolate missing values
        interp_points = np.zeros((len(missing_timepoints), x_values.shape[1]))

        for ii in range(x_values.shape[1]):
            f = interp1d(timepoints, x_values[:, ii], kind='linear', fill_value="extrapolate")
            interp_points[:, ii] = np.ceil(f(missing_timepoints))

        # Combine interpolated points with time column
        interp_points = np.column_stack((missing_timepoints, interp_points))

        # Concatenate original and interpolated data
        x = np.vstack((x, interp_points))

        # Sort by time (column 0)
        x = x[np.argsort(x[:, 0])]
        
        x_values = x[:, 1:]
        
        return x, x_values
        
        