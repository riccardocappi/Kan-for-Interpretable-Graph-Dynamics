from .Experiments import Experiments
from models.utils.MPNN import MPNN
from models.baseline.MPNN_ODE import MPNN_ODE
import torch
import torch.nn.functional as F
from models.utils.MLP import MLP

# Possible activation functions
activations = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus,
    "tanh": F.tanh,
    "identity": torch.nn.Identity()
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
        snr_db = -1,
        **kwargs
    ):
        super().__init__(config, n_trials, model_selection_method, study_name=study_name, process_id=process_id, 
                         snr_db=snr_db, **kwargs)
        
        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
        
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
        include_time = self.config.get("include_time", False)
        time_dim = 1 if include_time else 0
        
        in_dim = self.config.get('in_dim', 1)        
        if net_suffix == self.g_net_suffix:
            in_dim_ = 2 * in_dim
        elif (net_suffix == self.h_net_suffix) and message_passing:
            in_dim_ = 2 * in_dim + time_dim # Temporal component
        else:
            in_dim_ = in_dim + time_dim
        
        hidden_layers = [in_dim_] + hidden_layers + [in_dim]
        
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
            message_passing=self.config.get("message_passing", True),
            include_time=self.config.get("include_time", False)
        )
        
        model = MPNN_ODE(
            conv = net,
            model_path=f'{self.model_path}/mpnn',
            adjoint=self.config.get('adjoint', False),  # Should be read from config
            integration_method=self.integration_method,
            atol=self.config.get('atol', 1e-6), 
            rtol=self.config.get('rtol', 1e-3),
            predict_deriv=self.predict_deriv
        )
        model = model.to(torch.device(self.device))
        
        return model