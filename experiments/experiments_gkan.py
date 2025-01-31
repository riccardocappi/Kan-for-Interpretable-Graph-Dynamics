from .Experiments import Experiments
import torch
from models.utils.NetWrapper import NetWrapper
from models.GKAN_ODE import GKAN_ODE
from utils.utils import sample_from_spatio_temporal_graph, plot, save_acts


class ExperimentsGKAN(Experiments):
    def __init__(self, config, G, n_trials, model_selection_method='optuna'):
            
        super().__init__(config, G, n_trials, model_selection_method)
        
    
    def pre_processing(self, training_set, valid_set):
        return training_set, valid_set
    
    
    def get_model_opt(self, trial):
            
        grid_size = trial.suggest_int('grid_size', self.search_space['grid_size'][0],
                                      self.search_space['grid_size'][-1])
        
        spline_order = trial.suggest_int('spline_order', self.search_space['spline_order'][0], 
                                         self.search_space['spline_order'][-1])
        
        range_limit = trial.suggest_int('range_limit', self.search_space['range_limit'][0], 
                                        self.search_space['range_limit'][-1])
        
        grid_range = [-range_limit, range_limit]
                     
        lmbd_g = trial.suggest_float('lmbd_g', self.search_space['lmbd_g'][0], self.search_space['lmbd_g'][-1]) if self.use_reg_loss else 0.
        lmbd_h = trial.suggest_float('lmbd_h', self.search_space['lmbd_h'][0], self.search_space['lmbd_h'][-1]) if self.use_reg_loss else 0.
        is_lamb = lmbd_g > 0. or lmbd_h > 0.
        
        mu_1 = trial.suggest_float('mu_1', self.search_space['mu_1'][0], self.search_space['mu_1'][-1]) if self.use_reg_loss else 1.
        mu_2 = trial.suggest_float('mu_2', self.search_space['mu_2'][0], self.search_space['mu_2'][-1]) if self.use_reg_loss else 1.
            
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.search_space['use_orig_reg'])
        
        store_acts = (use_orig_reg and is_lamb)
        
        net = GKAN_ODE(
            h_hidden_layers = [2, 3, 1],
            g_hidden_layers = [2, 3, 1],
            grid_size = grid_size,
            spline_order = spline_order,
            grid_range = grid_range,
            model_path = self.model_path,
            store_acts = store_acts,
            device = self.device,
            mu_1 = mu_1,
            mu_2 = mu_2,
            use_orig_reg = store_acts,
            lmbd_g = lmbd_g,
            lmbd_h = lmbd_h
        )
        
        model = NetWrapper(net, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        return model
    
    
    
    def get_best_model(self, best_params):
        is_lamb = best_params['lmbd_g'] > 0. or best_params['lmbd_h'] > 0.
        store_acts = (best_params['use_orig_reg'] and is_lamb)
        range_limit = best_params['range_limit']
        grid_range = [-range_limit, range_limit]
        
        net = GKAN_ODE(
            h_hidden_layers = [2, 3, 1],
            g_hidden_layers = [2, 3, 1],
            grid_size = best_params['grid_size'],
            spline_order = best_params['spline_order'],
            grid_range = grid_range,
            model_path = f'{self.model_path}/eval',
            store_acts = store_acts,
            device = self.device,
            mu_1 = best_params.get('mu_1', 1.),
            mu_2 = best_params.get('mu_2', 1.),
            use_orig_reg = store_acts,
            lmbd_g = best_params.get('lmbd_g', 0.),
            lmbd_h = best_params.get('lmbd_h', 0.)
        )

        model = NetWrapper(net, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        return model

     
    def post_processing(self, best_model):
        
        net = best_model.model
        net.h_net.store_act = True
        net.g_net.store_act = True

        dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(self.training_set.data[0], 
                                                                    self.edge_index, 
                                                                    sample_size=32)

        with torch.no_grad():
            _ = net(dummy_x, dummy_edge_index)

        plot(folder_path=f'{net.h_net.model_path}/figures', layers=net.h_net.layers, show_plots=False)
        plot(folder_path=f'{net.g_net.model_path}/figures', layers=net.g_net.layers, show_plots=False)

        save_acts(layers=net.h_net.layers, folder_path=f'{net.h_net.model_path}/cached_acts')
        save_acts(layers=net.g_net.layers, folder_path=f'{net.g_net.model_path}/cached_acts') 