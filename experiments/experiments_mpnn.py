from .Experiments import Experiments
from models.baseline.baseline import MPNN
from models.utils.NetWrapper import NetWrapper
import torch
import torch.nn.functional as F
from models.baseline.baseline import MLP

# Possible activation functions
activations = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus
}


class ExperimentsMPNN(Experiments):
    """
    Implements the experiment pipeline for Message Passing Neural Network-ODE (MPNN-ODE) model
    """
    def __init__(
        self, 
        config,
        n_trials, 
        model_selection_method='optuna',
        study_name = 'example',
        process_id = 0,
        **kwargs
    ):
        super().__init__(config, n_trials, model_selection_method, study_name=study_name, process_id=process_id, **kwargs)
        
        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data   # No pre-processing
    
    
    def _get_mlp_config_trial(self, trial, net_suffix):
        """
        Returns the configuration of a single MLP
        
        Args:
            - trial : current optuna trial
            - net_suffix : whether to consider the g_net hyper-params or the h_net ones
        """
        n_hidden_layers = trial.suggest_int(
            f'n_hidden_layers_{net_suffix}', 
            self.search_space[f'n_hidden_layers_{net_suffix}'][0], 
            self.search_space[f'n_hidden_layers_{net_suffix}'][-1]
        )
        
        hidden_dims = trial.suggest_int(
            f'hidden_dims_{net_suffix}',
            self.search_space[f'hidden_dims_{net_suffix}'][0],
            self.search_space[f'hidden_dims_{net_suffix}'][-1],
            step=8
        )
        
        hidden_layers = [hidden_dims for _ in range(n_hidden_layers)]
        
        message_passing = self.config.get("message_passing", True)
        
        in_dims = 2*self.config['in_dim'] if message_passing or (net_suffix == self.g_net_suffix) else self.config["in_dim"] 
        
        hidden_layers = [in_dims] + hidden_layers + [self.config['in_dim']]
        
        activation = trial.suggest_categorical(
            f'af_{net_suffix}',
            self.search_space[f'af_{net_suffix}']
        )
        
        af = activations[activation]
        
        dropout_rate = trial.suggest_float(
            f'drop_p_{net_suffix}',
            self.search_space[f'drop_p_{net_suffix}'][0],
            self.search_space[f'drop_p_{net_suffix}'][-1],
            log=True
        )
        
        mlp_config = {
            'hidden_layers': hidden_layers,
            'af': af,
            'model_path': f'{self.model_path}/{net_suffix}',
            'dropout_rate': dropout_rate
        }
        
        return mlp_config
    
    
    def get_model_opt(self, trial):
        """
        Constructs MPNN model
        """
        g_net_config = self._get_mlp_config_trial(trial=trial, net_suffix=self.g_net_suffix)
        h_net_config = self._get_mlp_config_trial(trial=trial, net_suffix=self.h_net_suffix)
        
        g_net = MLP(**g_net_config)
        h_net = MLP(**h_net_config)
        
        net = MPNN(
            g_net=g_net,
            h_net=h_net,
            model_path=self.model_path,
            message_passing=self.config.get("message_passing", True)
        )
        
        model = NetWrapper(net, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model