from .Experiments import Experiments
import torch
from models.utils.NetWrapper import NetWrapper
from models.GKAN import GKAN
from models.kan.KAN import KAN



class ExperimentsGKAN(Experiments):
    """
    Implements the experiment pipeline for GKAN-ODE models
    """
    def __init__(
        self, 
        config, 
        G, 
        n_trials, 
        model_selection_method='optuna',
        study_name = 'example',
        process_id=0
    ):
        super().__init__(config, G, n_trials, model_selection_method, study_name=study_name, process_id=process_id)

        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
    
    def pre_processing(self, training_set, valid_set):
        return training_set, valid_set  # No pre-processing
    
    
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
        
        mu_1 = 1.
        mu_2 = 1.
                    
        hidden_dim = trial.suggest_int(f'hidden_dim_{net_suffix}', self.search_space[f'hidden_dim_{net_suffix}'][0], 
                                       self.search_space[f'hidden_dim_{net_suffix}'][-1])
        
        in_dim = self.config['in_dim']
        
        message_passing = self.config.get("message_passing", True)  # Whether to use the message_passing definition or not
        
        if message_passing or net_suffix == self.g_net_suffix:
            hidden_layers = [2*in_dim, hidden_dim, in_dim]
        else:
            hidden_layers = [in_dim, hidden_dim, in_dim]    # If the model is defined as H(x_i) + \sum_{j} G(x_i, x_j), the H net takes as input only x_i feature vector
        
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
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.search_space['use_orig_reg'])
        
        # regularization parameter for the G net
        lamb_g = trial.suggest_float(
            f'lamb_{self.g_net_suffix}', 
            self.search_space[f'lamb_{self.g_net_suffix}'][0], 
            self.search_space[f'lamb_{self.g_net_suffix}'][-1],
            step = 0.0001
        )
        
        # regularization parameter for the H net
        lamb_h = trial.suggest_float(f'lamb_{self.h_net_suffix}', 
                                     self.search_space[f'lamb_{self.h_net_suffix}'][0], 
                                     self.search_space[f'lamb_{self.h_net_suffix}'][-1],
                                     step = 0.0001)
                
        
        g_net_config = self._get_kan_trial_config(trial, net_suffix=self.g_net_suffix, use_orig_reg=(use_orig_reg and lamb_g > 0.))
        h_net_config = self._get_kan_trial_config(trial, net_suffix=self.h_net_suffix, use_orig_reg=(use_orig_reg and lamb_h > 0.))
        
        g_net = KAN(**g_net_config)
        h_net = KAN(**h_net_config)        
        
        net = GKAN(
            h_net=h_net,
            g_net=g_net,
            model_path = f"{self.model_path}/gkan",
            device = self.device,
            lmbd_g=lamb_g,
            lmbd_h=lamb_h,
            message_passing=self.config.get("message_passing", True)
        )
        
        model = NetWrapper(net, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        return model