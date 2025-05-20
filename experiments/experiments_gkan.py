from .Experiments import Experiments
import torch
from models.GKAN_ODE import GKAN_ODE
from models.utils.MPNN import MPNN
from models.kan.KAN import KAN


class ExperimentsGKAN(Experiments):
    """
    Implements the experiment pipeline for GKAN-ODE models
    """
    def __init__(
        self, 
        config, 
        n_trials, 
        model_selection_method='optuna',
        study_name = 'example',
        process_id=0,
        **kwargs
    ):
        super().__init__(config, n_trials, model_selection_method, study_name=study_name, process_id=process_id, **kwargs)

        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
            
    def _get_kan_trial_config(self, trial, net_suffix, use_orig_reg):
        
        """
        Returns the configuration of a single KAN
        
        Args:
            - trial : current optuna trial
            - net_suffix : whether to consider the g_net hyper-params or the h_net ones
            - use_orig_reg : whether to use the original regularization loss or the one defined by efficient_kan 
        """
        grid_size = trial.suggest_int(f'grid_size_{net_suffix}', self.search_space[f'grid_size_{net_suffix}'][0],
                                      self.search_space[f'grid_size_{net_suffix}'][-1])
        
        spline_order = trial.suggest_int(f'spline_order_{net_suffix}', self.search_space[f'spline_order_{net_suffix}'][0], 
                                         self.search_space[f'spline_order_{net_suffix}'][-1])
        
        range_limit = trial.suggest_int(f'range_limit_{net_suffix}', self.search_space[f'range_limit_{net_suffix}'][0], 
                                        self.search_space[f'range_limit_{net_suffix}'][-1])
        
        grid_range = [-range_limit, range_limit]
        
        mu_1 = trial.suggest_float(f'mu_1_{net_suffix}', self.search_space[f'mu_1_{net_suffix}'][0], 
                                   self.search_space[f'mu_1_{net_suffix}'][-1], step=0.1)
        
        mu_2 = trial.suggest_float(f'mu_2_{net_suffix}', self.search_space[f'mu_2_{net_suffix}'][0], 
                                   self.search_space[f'mu_2_{net_suffix}'][-1], step=0.1)
                    
        hidden_dim = trial.suggest_int(f'hidden_dim_{net_suffix}', self.search_space[f'hidden_dim_{net_suffix}'][0], 
                                       self.search_space[f'hidden_dim_{net_suffix}'][-1])
        
        in_dim = self.config.get('in_dim', 1)
        
        message_passing = self.config.get("message_passing", True)  # Whether to use the message_passing definition or not
        include_time = self.config.get("include_time", False)
        time_dim = 1 if include_time else 0
        
        in_dim = self.config.get('in_dim', 1)        
        if net_suffix == self.g_net_suffix:
            in_dim_ = 2 * in_dim
        elif (net_suffix == self.h_net_suffix) and message_passing:
            in_dim_ = 2 * in_dim + time_dim # Temporal component
        else:
            in_dim_ = in_dim + time_dim
            
        hidden_layers = [in_dim_, hidden_dim, in_dim]
        
        kan_config = {
            'layers_hidden':hidden_layers,
            'grid_size':grid_size,
            'spline_order':spline_order,
            'grid_range':grid_range,
            'store_act':use_orig_reg,
            'device':self.device,
            'mu_1':mu_1,
            'mu_2':mu_2,
            'use_orig_reg':use_orig_reg
        }
        
        return kan_config
        

    def get_model_opt(self, trial):
        """
        Construct GKAN-ODE model
        
        Args:
            -trial : current optuna trial
        """
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.search_space["use_orig_reg"])
        
        # regularization parameter for the G net
        lamb_g = trial.suggest_float(
            f'lamb_{self.g_net_suffix}', 
            self.search_space[f'lamb_{self.g_net_suffix}'][0], 
            self.search_space[f'lamb_{self.g_net_suffix}'][-1],
            log=True
        )
        
        # regularization parameter for the H net
        lamb_h = trial.suggest_float(
            f'lamb_{self.h_net_suffix}',                          
            self.search_space[f'lamb_{self.h_net_suffix}'][0], 
            self.search_space[f'lamb_{self.h_net_suffix}'][-1],
            log=True
        )
        
        g_net_config = self._get_kan_trial_config(trial, net_suffix=self.g_net_suffix, use_orig_reg=(use_orig_reg and lamb_g > 0.))
        h_net_config = self._get_kan_trial_config(trial, net_suffix=self.h_net_suffix, use_orig_reg=(use_orig_reg and lamb_h > 0.))
        
        g_net = KAN(**g_net_config)
        h_net = KAN(**h_net_config)        
        
        conv = MPNN(
            h_net=h_net,
            g_net=g_net,
            message_passing=self.config.get("message_passing", True),
            include_time=self.config.get("include_time", True)
        )
        
        model = GKAN_ODE(
            conv = conv,
            model_path = f"{self.model_path}/gkan",
            adjoint=self.config.get('adjoint', False),
            integration_method=self.integration_method,
            lmbd_g=lamb_g,
            lmbd_h=lamb_h,
            atol=self.config.get('atol', 1e-6), 
            rtol=self.config.get('rtol', 1e-3),
            predict_deriv=self.predict_deriv
        )
        model = model.to(torch.device(self.device))
        
        return model