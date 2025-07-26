from .Experiments import Experiments
from models.utils.MPNN import MPNN
from models.baseline.LLC import LLC_ODE
import torch
import torch.nn.functional as F
from models.baseline.LLC_Conv import Q_inter, Q_self
from experiments.experiments_mpnn import activations

class ExperimentsLLC(Experiments):
    def __init__(
        self, 
        config, 
        n_trials, 
        model_selection_method='optuna', 
        study_name='example', 
        process_id=0, 
        snr_db=-1,
        denoise=False,
        **kwargs
    ):
        super().__init__(config, n_trials, model_selection_method, study_name, process_id, snr_db, denoise, **kwargs)
        self.g_net_suffix = "g0"
        self.h_net_suffix = "h_net"
    
    
    def __get_mlp_config_trial(self, trial, net_suffix):

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
        
        if net_suffix is not self.h_net_suffix:
            mlp_config = {
                f'hidden_layers_{net_suffix}': hidden_layers,
                f'af_{net_suffix}': af,
                f'dr_{net_suffix}': dropout_rate
            }
        else:
            mlp_config = {
                'hidden_layers': hidden_layers,
                'af': af,
                'dropout_rate': dropout_rate
            }
        
        return mlp_config
    
    
    def get_model_opt(self, trial):
        g0_config = self.__get_mlp_config_trial(trial=trial, net_suffix=self.g_net_suffix)
        g1_config = self.__get_mlp_config_trial(trial=trial, net_suffix="g1")
        g2_config = self.__get_mlp_config_trial(trial=trial, net_suffix="g2")
        
        g_net_config = {**g0_config, **g1_config, **g2_config}
        h_net_config = self.__get_mlp_config_trial(trial=trial, net_suffix=self.h_net_suffix)
        
        g_net = Q_inter(**g_net_config)
        h_net = Q_self(**h_net_config)
        
        net = MPNN(
            g_net=g_net,
            h_net=h_net,
            message_passing=self.config.get("message_passing", True),
            include_time=self.config.get("include_time", False)
        )
        
        model = LLC_ODE(
            conv=net,
            model_path=f'{self.model_path}/llc',
            adjoint=self.config.get('adjoint', False),
            integration_method=self.integration_method,
            predict_deriv=self.predict_deriv,
            atol=self.config.get('atol', 1e-6), 
            rtol=self.config.get('rtol', 1e-3)
        )
        
        model = model.to(torch.device(self.device))
        return model
        
        
        
        
    
        