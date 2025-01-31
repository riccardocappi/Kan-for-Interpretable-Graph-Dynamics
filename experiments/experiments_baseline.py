from .Experiments import Experiments
from models.utils.NetWrapper import NetWrapper
from models.baseline.baseline import GCN, GIN
import torch
from utils.utils import sample_from_spatio_temporal_graph
import os


class ExperimentsBaseline(Experiments):
    
    def __init__(self, config, G, n_trials, model_selection_method='optuna', model_type='GCN'):
        super().__init__(config, G, n_trials, model_selection_method)
        self.model_type = model_type
    
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data
    
    
    def get_model_opt(self, trial):
        if self.model_type == 'GCN':
            model = self._get_GCN(trial)
        elif self.model_type == 'GIN':
            model = self._get_GIN(trial)
        else:
            raise Exception('Model not supported')

        return model
        

    def _get_GCN(self, trial):
        hidden_dims = trial.suggest_int('hidden_dim', self.search_space['hidden_dim'][0], self.search_space['hidden_dim'][-1])
        net = GCN(
            input_dim = self.config['in_dim'],
            hidden_dim = hidden_dims,
            output_dim = self.config['in_dim'],
            model_path = self.model_path
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model
    
    
    def _get_GIN(self, trial):
        hidden_dims = trial.suggest_int('hidden_dim', self.search_space['hidden_dim'][0], self.search_space['hidden_dim'][-1])
        epsilon = trial.suggest_float('epsilon', self.search_space['epsilon'][0], self.search_space['epsilon'][-1])
        
        net = GIN(
            input_dim = self.config['in_dim'],
            hidden_dim = hidden_dims,
            output_dim = self.config['in_dim'],
            model_path = self.model_path,
            epsilon = epsilon
        )
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model

    
    def get_best_model(self, best_params):
        model_config = {}
        for key in self.search_space.keys():
            if key != 'lr':
                model_config[key] = best_params[key]
            
        model_config['model_path'] = f'{self.model_path}/eval'
        model_config['input_dim'] = self.config['in_dim']
        model_config['output_dim'] = self.config['in_dim']
        
        net = GCN if self.model_type == 'GCN' else GIN          #TODO: Generalize this
        net_instance = net(**model_config)
        
        model = NetWrapper(net_instance, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model

    
    def post_processing(self, best_model):
        
        net = best_model.model
        
        net.save_black_box = True
        dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(self.training_set.data[0], 
                                                                      self.edge_index, 
                                                                      sample_size=32)
        
        with torch.no_grad():
            _ = net(dummy_x, dummy_edge_index)
            
        folder_path = f'{net.model_path}/cached_data'
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        torch.save(net.cache_input, f'{folder_path}/cached_input')
        torch.save(net.cache_output, f'{folder_path}/cached_output')
        