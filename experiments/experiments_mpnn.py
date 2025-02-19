from .Experiments import Experiments
from models.baseline.baseline import MPNN
from models.utils.NetWrapper import NetWrapper
import torch
import torch.nn.functional as F
from models.baseline.baseline import MLP


class ExperimentsMPNN(Experiments):
    def __init__(
        self, 
        config, 
        G, 
        n_trials, 
        model_selection_method='optuna',
        study_name = 'example',
        eval_model=True,
        process_id = 0
    ):
        super().__init__(config, G, n_trials, model_selection_method, study_name=study_name,eval_model=eval_model, process_id=process_id)
        
        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data
    
    
    def _get_mlp_config_trial(self, trial, net_suffix):
        n_hidden_layers = trial.suggest_int(
            f'n_hidden_layers_{net_suffix}', 
            self.search_space[f'n_hidden_layers_{net_suffix}'][0], 
            self.search_space[f'n_hidden_layers_{net_suffix}'][-1]
        )
        
        hidden_dims = trial.suggest_int(
            f'hidden_dims_{net_suffix}',
            self.search_space[f'hidden_dims_{net_suffix}'][0],
            self.search_space[f'hidden_dims_{net_suffix}'][-1]
        )
        
        hidden_layers = [hidden_dims for _ in range(n_hidden_layers)]
        hidden_layers = [2*self.config['in_dim']] + hidden_layers + [self.config['in_dim']]
        
        mlp_config = {
            'hidden_layers': hidden_layers,
            'af': F.relu,
            'model_path': f'{self.model_path}/{net_suffix}'
        }
        
        return mlp_config
    
    
    def get_model_opt(self, trial):
        
        g_net_config = self._get_mlp_config_trial(trial=trial, net_suffix=self.g_net_suffix)
        h_net_config = self._get_mlp_config_trial(trial=trial, net_suffix=self.h_net_suffix)
        
        g_net = MLP(**g_net_config)
        h_net = MLP(**h_net_config)
        
        net = MPNN(
            g_net=g_net,
            h_net=h_net,
            model_path=self.model_path
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model
        
    
    
    def _get_best_mlp_config(self, best_params, net_suffix):
        n_hidden_layers = best_params[f'n_hidden_layers_{net_suffix}']
        hidden_dims = best_params[f'hidden_dims_{net_suffix}']
        
        hidden_layers = [hidden_dims for _ in range(n_hidden_layers)]
        hidden_layers = [2*self.config['in_dim']] + hidden_layers + [self.config['in_dim']]
        
        mlp_config = {
            'hidden_layers': hidden_layers,
            'af': F.relu,
            'model_path': f'{self.model_path}/eval/{net_suffix}'
        }
        
        return mlp_config
    
    
    def get_best_model(self, best_params):
        
        g_net_config = self._get_best_mlp_config(best_params=best_params, net_suffix=self.g_net_suffix)
        h_net_config = self._get_best_mlp_config(best_params=best_params, net_suffix=self.h_net_suffix)
        
        g_net = MLP(**g_net_config)
        h_net = MLP(**h_net_config)
        
        net = MPNN(
            g_net=g_net,
            h_net=h_net,
            model_path=f'{self.model_path}/eval'
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model