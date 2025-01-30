from .Experiments import Experiments
import torch
from models.utils.NetWrapper import NetWrapper
from models.GKAN_ODE import GKAN_ODE
from train_and_eval import fit
from utils.utils import sample_from_spatio_temporal_graph, plot, save_acts


class ExperimentsGKAN(Experiments):
    def __init__(self, config, G, n_trials, model_selection_method='optuna', t_f_train=240):
        
        search_space = None
        if model_selection_method == 'grid_search':
            search_space = {
                'grid_size': config['grid_size'],
                'spline_order': config['spline_order'],
                'range_limit': config['range_limit'],
                'lr': config['lr'],
                'lmbd_g': config['lmbd_g'],
                'lmbd_h': config['lmbd_h'],
                'mu_1': config['mu_1'],
                'mu_2': config['mu_2'],
                'use_orig_reg': config['use_orig_reg']
            }
            
        super().__init__(config, G, n_trials, search_space, model_selection_method, t_f_train=t_f_train)
        
    
    def pre_processing(self, training_set, valid_set):
        return training_set, valid_set
    
    
    def objective(self, trial):
            
        grid_size = trial.suggest_int('grid_size', self.config['grid_size'][0],
                                      self.config['grid_size'][-1])
        
        spline_order = trial.suggest_int('spline_order', self.config['spline_order'][0], 
                                         self.config['spline_order'][-1])
        
        range_limit = trial.suggest_int('range_limit', self.config['range_limit'][0], 
                                        self.config['range_limit'][-1])
        
        grid_range = [-range_limit, range_limit]
             
        lr = trial.suggest_float('lr', self.config['lr'][0], self.config['lr'][-1])
        
        lmbd_g = trial.suggest_float('lmbd_g', self.config['lmbd_g'][0], self.config['lmbd_g'][-1]) if self.use_reg_loss else 0.
        lmbd_h = trial.suggest_float('lmbd_h', self.config['lmbd_h'][0], self.config['lmbd_h'][-1]) if self.use_reg_loss else 0.
        is_lamb = lmbd_g > 0. or lmbd_h > 0.
        
        mu_1 = trial.suggest_float('mu_1', self.config['mu_1'][0], self.config['mu_1'][-1]) if self.use_reg_loss else 1.
        mu_2 = trial.suggest_float('mu_2', self.config['mu_2'][0], self.config['mu_2'][-1]) if self.use_reg_loss else 1.
            
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.config['use_orig_reg'])
        
        store_acts = (use_orig_reg and is_lamb)
        
        model_config = {
            'h_hidden_layers': [2, 3, 1],
            'g_hidden_layers': [2, 3, 1],
            'grid_size': grid_size,
            'spline_order': spline_order,
            'grid_range': grid_range,
            'model_path': self.model_path,
            'store_acts': store_acts,
            'device': self.device,
            'mu_1': mu_1,
            'mu_2': mu_2,
            'use_orig_reg': store_acts,
            'lmbd_g': lmbd_g,
            'lmbd_h': lmbd_h
        }
        
        model = NetWrapper(GKAN_ODE, model_config, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        results = fit(
            model,
            self.training_set,
            self.valid_set,
            epochs=self.epochs,
            patience=self.patience,
            lr = lr,
            lmbd=1.,
            log=self.log,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            save_updates=False,
            n_iter=self.n_iter,
            batch_size=-1,
            t_f_train=self.t_f_train
        )
        
        best_val_loss = min(results['validation_loss'])
        
        return best_val_loss
    
    
    
    def eval_model(self, best_params):
        is_lamb = best_params['lmbd_g'] > 0. or best_params['lmbd_h'] > 0.
        store_acts = (best_params['use_orig_reg'] and is_lamb)
        range_limit = best_params['range_limit']
        grid_range = [-range_limit, range_limit]
        
        model_config = {
            'h_hidden_layers': [2, 3, 1],
            'g_hidden_layers': [2, 3, 1],
            'grid_size': best_params['grid_size'],
            'spline_order': best_params['spline_order'],
            'grid_range': grid_range,
            'model_path': f'{self.model_path}/eval',
            'store_acts': store_acts,
            'device': self.device,
            'mu_1': best_params.get('mu_1', 1.),
            'mu_2': best_params.get('mu_2', 1.),
            'use_orig_reg': store_acts,
            'lmbd_g': best_params['lmbd_g'],
            'lmbd_h': best_params['lmbd_h']
        }

        model = NetWrapper(GKAN_ODE, model_config, self.edge_index, update_grid=False)
        model = model.to(torch.device(self.device))
        
        _ = fit(
            model,
            self.training_set,
            self.valid_set,
            epochs=self.epochs,
            patience=self.patience,
            lr = best_params['lr'],
            lmbd=1.,
            log=self.log,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            save_updates=True,
            n_iter=self.n_iter,
            batch_size=-1,
            t_f_train=self.t_f_train
        ) 
        
        return model

     
    def post_processing(self, best_params):
        model = self.eval_model(best_params)
        
        net = model.model
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