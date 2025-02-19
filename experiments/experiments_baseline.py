from .Experiments import Experiments
from models.utils.NetWrapper import NetWrapper
from models.baseline.baseline import GCN, GIN
import torch


class ExperimentsBaseline(Experiments):
    
    def __init__(
        self,
        config, 
        G, 
        n_trials, 
        model_selection_method='optuna', 
        model_type='GCN',
        study_name='example',
        eval_model=True,
        process_id = 0
    ):
        super().__init__(config, G, n_trials, model_selection_method, study_name=study_name, eval_model=eval_model, process_id=process_id)
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
        