from .Experiments import Experiments
import torch
from models.utils.NetWrapper import NetWrapper
from models.GKAN_ODE import GKAN_ODE
from models.kan.KAN import KAN



class ExperimentsGKAN(Experiments):
    def __init__(
        self, 
        config, 
        G, 
        n_trials, 
        model_selection_method='optuna',
        study_name = 'example',
        eval_model = True,
        process_id=0
    ):
        super().__init__(config, G, n_trials, model_selection_method, study_name=study_name, eval_model=eval_model, process_id=process_id)

        self.h_net_suffix = 'h_net'
        self.g_net_suffix = 'g_net'
        
    
    def pre_processing(self, training_set, valid_set):
        return training_set, valid_set
    
    
    def _get_kan_trial_config(self, trial, net_suffix, use_orig_reg):
        grid_size = trial.suggest_int(f'grid_size_{net_suffix}', self.search_space[f'grid_size_{net_suffix}'][0],
                                      self.search_space[f'grid_size_{net_suffix}'][-1])
        
        spline_order = trial.suggest_int(f'spline_order_{net_suffix}', self.search_space[f'spline_order_{net_suffix}'][0], 
                                         self.search_space[f'spline_order_{net_suffix}'][-1])
        
        range_limit = trial.suggest_int(f'range_limit_{net_suffix}', self.search_space[f'range_limit_{net_suffix}'][0], 
                                        self.search_space[f'range_limit_{net_suffix}'][-1])
        
        grid_range = [-range_limit, range_limit]
        
        # mu_1 = trial.suggest_float(f'mu_1_{net_suffix}', self.search_space['mu_1'][0], self.search_space['mu_1'][-1]) if self.use_reg_loss else 1.
        # mu_2 = trial.suggest_float(f'mu_2_{net_suffix}', self.search_space['mu_2'][0], self.search_space['mu_2'][-1]) if self.use_reg_loss else 1.
        mu_1 = 1.
        mu_2 = 1.
                
        hidden_dim = trial.suggest_int(f'hidden_dim_{net_suffix}', self.search_space[f'hidden_dim_{net_suffix}'][0], 
                                       self.search_space[f'hidden_dim_{net_suffix}'][-1])
        
        in_dim = self.config['in_dim']
        hidden_layers = [2*in_dim, hidden_dim, in_dim]
        
        model_path = f'{self.model_path}/{net_suffix}'
        
        kan_config = {
            'layers_hidden':hidden_layers,
            'grid_size':grid_size,
            'spline_order':spline_order,
            'grid_range':grid_range,
            'model_path':model_path,
            'store_act':use_orig_reg,
            'device':self.device,
            'mu_1':mu_1,
            'mu_2':mu_2,
            'use_orig_reg':use_orig_reg
        }
        
        return kan_config
        

    def get_model_opt(self, trial):
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.search_space['use_orig_reg'])
        
        lamb_g = trial.suggest_float(f'lamb_{self.g_net_suffix}', 
                                     self.search_space[f'lamb_{self.g_net_suffix}'][0], 
                                     self.search_space[f'lamb_{self.g_net_suffix}'][-1])
        
        lamb_h = trial.suggest_float(f'lamb_{self.h_net_suffix}', 
                                     self.search_space[f'lamb_{self.h_net_suffix}'][0], 
                                     self.search_space[f'lamb_{self.h_net_suffix}'][-1])
                
        
        g_net_config = self._get_kan_trial_config(trial, net_suffix=self.g_net_suffix, use_orig_reg=use_orig_reg)
        h_net_config = self._get_kan_trial_config(trial, net_suffix=self.h_net_suffix, use_orig_reg=use_orig_reg)
        
        g_net = KAN(**g_net_config)
        h_net = KAN(**h_net_config)        
        
        net = GKAN_ODE(
            h_net=h_net,
            g_net=g_net,
            model_path = self.model_path,
            device = self.device,
            lmbd_g=lamb_g,
            lmbd_h=lamb_h
        )
        model = NetWrapper(net, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        return model
    
    
    
    def _get_kan_best_config(self, best_params, net_suffix):
        hidden_layers = [2*self.config['in_dim'], best_params[f'hidden_dim_{net_suffix}'], self.config['in_dim']]
        range_limit = best_params[f'range_limit_{net_suffix}']
        grid_range = [-range_limit, range_limit]
        
        kan_config = {
            'layers_hidden':hidden_layers,
            'grid_size':best_params[f'grid_size_{net_suffix}'],
            'spline_order':best_params[f'spline_order_{net_suffix}'],
            'grid_range':grid_range,
            'model_path':f'{self.model_path}/eval/{net_suffix}',
            'store_act':best_params['use_orig_reg'],
            'device':self.device,
            'mu_1':best_params.get(f'mu_1_{net_suffix}', 1.),
            'mu_2':best_params.get(f'mu_2_{net_suffix}', 1.),
            'use_orig_reg':best_params['use_orig_reg']
        }
        
        return kan_config
         
    
    def get_best_model(self, best_params):
        
        g_net_config = self._get_kan_best_config(best_params=best_params, net_suffix=self.g_net_suffix)
        h_net_config = self._get_kan_best_config(best_params=best_params, net_suffix=self.h_net_suffix)
        
        g_net = KAN(**g_net_config)
        h_net = KAN(**h_net_config)
        
        net = GKAN_ODE(
            h_net=h_net,
            g_net=g_net,
            model_path=f'{self.model_path}/eval',
            device=self.device,
            lmbd_g=best_params[f'lamb_{self.g_net_suffix}'],
            lmbd_h=best_params[f'lamb_{self.h_net_suffix}']
        )

        model = NetWrapper(net, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        return model