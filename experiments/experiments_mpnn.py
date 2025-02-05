from .Experiments import Experiments
from models.baseline.baseline import MPNN
from models.utils.NetWrapper import NetWrapper
import torch


class ExperimentsMPNN(Experiments):
    def __init__(self, config, G, n_trials, model_selection_method='optuna'):
        super().__init__(config, G, n_trials, model_selection_method)
        
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data
    
    
    def get_model_opt(self, trial):
        
        n_hidden_layers = trial.suggest_int(
            'n_hidden_layers', 
            self.search_space['n_hidden_layers'][0], 
            self.search_space['n_hidden_layers'][-1]
        )
        
        hidden_dims = trial.suggest_int(
            'hidden_dims',
            self.search_space['hidden_dims'][0],
            self.search_space['hidden_dims'][-1]
        )
        
        hidden_layers = [hidden_dims for _ in range(n_hidden_layers)]
        hidden_layers = [2*self.config['in_dim']] + hidden_layers + [self.config['in_dim']] 
        
        net = MPNN(
            g_hidden_layers=hidden_layers,
            h_hidden_layers=hidden_layers,
            model_path=self.model_path
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model
        
    
    
    def get_best_model(self, best_params):
        n_hidden_layers = best_params['n_hidden_layers']
        hidden_dims = best_params['hidden_dims']
        
        hidden_layers = [hidden_dims for _ in range(n_hidden_layers)]
        hidden_layers = [2*self.config['in_dim']] + hidden_layers + [self.config['in_dim']]
        
        net = MPNN(
            g_hidden_layers=hidden_layers,
            h_hidden_layers=hidden_layers,
            model_path=f'{self.model_path}/eval'
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model
        
        
        