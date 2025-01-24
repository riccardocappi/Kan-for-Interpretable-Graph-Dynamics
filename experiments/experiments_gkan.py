from .Experiments import Experiments
import torch
from models.NetWrapper import NetWrapper
from models.KanGDyn import KanGDyn
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
                'lamb': config['lamb'],
                'mu_1': config['mu_1'],
                'mu_2': config['mu_2'],
                'use_orig_reg': config['use_orig_reg']
            }
            
        super().__init__(config, G, n_trials, search_space, model_selection_method, t_f_train=t_f_train)
        
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data
    
    
    def objective(self, trial):
            
        grid_size = trial.suggest_int('grid_size', self.config['grid_size'][0],
                                      self.config['grid_size'][-1])
        
        spline_order = trial.suggest_int('spline_order', self.config['spline_order'][0], 
                                         self.config['spline_order'][-1])
        
        range_limit = trial.suggest_int('range_limit', self.config['range_limit'][0], 
                                        self.config['range_limit'][-1])
        
        grid_range = [-range_limit, range_limit]
             
        lr = trial.suggest_float('lr', self.config['lr'][0], self.config['lr'][-1])
        
        lamb = trial.suggest_float('lamb', self.config['lamb'][0], self.config['lamb'][-1]) if self.use_reg_loss else 0. 
        mu_1 = trial.suggest_float('mu_1', self.config['mu_1'][0], self.config['mu_1'][-1]) if self.use_reg_loss else 1.
        mu_2 = trial.suggest_float('mu_2', self.config['mu_2'][0], self.config['mu_2'][-1]) if self.use_reg_loss else 1.
            
        use_orig_reg = trial.suggest_categorical("use_orig_reg", self.config['use_orig_reg'])
        
        store_acts = (use_orig_reg and lamb > 0.)
        
        model_config = {
            'h_hidden_layers': [2, 3, 1],
            'g_hidden_layers': [2, 3, 1],
            'grid_size': grid_size,
            'spline_order': spline_order,
            'grid_range': grid_range,
            'model_path': self.model_path,
            'store_acts': store_acts,
            'device': self.device
        }
        
        model = NetWrapper(KanGDyn, model_config, self.edge_index, update_grid=False)
        model.to(torch.device(self.device))
        
        results = fit(
            model,
            self.train_data,
            self.t_train,
            self.valid_data,
            self.t_valid,
            epochs=self.epochs,
            patience=self.patience,
            lr = lr,
            lmbd=lamb,
            log=self.log,
            mu_1=mu_1,
            mu_2=mu_2,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            use_orig_reg=store_acts,
            save_updates=False,
            t_f_train=self.t_f_train,
            n_iter=self.n_iter
        )
        
        best_val_loss = min(results['validation_loss'])
        
        return best_val_loss
    
    
    
    def eval_model(self, best_params):
        store_acts = (best_params['use_orig_reg'] and best_params['lamb'] > 0.)
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
            'device': self.device
        }

        model = NetWrapper(KanGDyn, model_config, self.edge_index, update_grid=False)
        model.to(torch.device(self.device))
        
        _ = fit(
            model,
            self.train_data,
            self.t_train,
            self.valid_data,
            self.t_valid,
            epochs=self.epochs,
            patience=self.patience,
            lr = best_params['lr'],
            lmbd=best_params.get('lamb', 0.),
            log=self.log,
            mu_1=best_params.get('mu_1', 1.),
            mu_2=best_params.get('mu_2', 1.),
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            use_orig_reg=store_acts,
            save_updates=True,
            t_f_train=self.t_f_train,
            n_iter=self.n_iter
        ) 
        
        return model

     
    def post_processing(self, best_params):
        model = self.eval_model(best_params)
        
        net = model.model
        net.h_net.store_act = True
        net.g_net.store_act = True

        dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(self.train_data[0], 
                                                                    self.edge_index, 
                                                                    sample_size=32)

        with torch.no_grad():
            _ = net(dummy_x, dummy_edge_index)

        plot(folder_path=f'{net.h_net.model_path}/figures', layers=net.h_net.layers, show_plots=False)
        plot(folder_path=f'{net.g_net.model_path}/figures', layers=net.g_net.layers, show_plots=False)

        save_acts(layers=net.h_net.layers, folder_path=f'{net.h_net.model_path}/cached_acts')
        save_acts(layers=net.g_net.layers, folder_path=f'{net.g_net.model_path}/cached_acts') 