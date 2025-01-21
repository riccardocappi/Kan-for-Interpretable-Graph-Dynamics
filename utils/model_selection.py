import torch
import numpy as np
from .utils import create_datasets, pre_processing
from train_and_eval import fit
import optuna
from optuna.samplers import GridSampler
import json
from models.NetWrapper import NetWrapper
from models.KanGDyn import KanGDyn
from torch_geometric.utils import from_networkx



class ModelSelector():
    
    def __init__(self, config, G, noise_level=None, n_trials=None, method='grid_search'):
        
        super().__init__()
        
        assert method != 'optuna' or n_trials is not None
        assert method == 'optuna' or method == 'grid_search', 'Optimization method not supported!'
        
        self.n_trials = n_trials
        self.method = method

        self.device = config["device"]
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            
        self.train_data, self.t_train, self.valid_data, self.t_valid, self.test_data, self.t_test = create_datasets(config, G)
        
        self.train_data = pre_processing(self.train_data)
        self.valid_data = pre_processing(self.valid_data)
        self.test_data = pre_processing(self.test_data)
        
        self.edge_index = from_networkx(G).edge_index
        self.edge_index = self.edge_index.to(torch.device(self.device))
        
        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        # self.input_noise = config["add_noise_input"]
        
        self.model_path = f'./saved_models_optuna/{config["model_name"]}'
        self.use_reg_loss = config['reg_loss']
        
        # Model hyper-params
        self.model_config = {
            'h_hidden_layers': config['h_hidden_layers'],
            'g_hidden_layers': config['g_hidden_layers'],
            'model_path': self.model_path,
            'device': self.device
        }
        
        
    
    def optimize(self):
        # Maybe get search_space from config
        if self.method == 'grid_search':
            search_space = {
                'grid_size': [5, 7],
                'range_limit': [3, 5],
                'spline_order': [3],
                'lr': [0.01, 0.005],
                'lamb': [0., 0.0001] if self.use_reg_loss else [0.],
                'mu_1': [1.] if self.use_reg_loss else [1.],
                'mu_2': [1.] if self.use_reg_loss else [1.],
                'use_orig_reg': [True, False]
            }
            sampler = GridSampler(search_space)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(self.objective, n_trials=len(sampler._all_grids))
        
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=self.n_trials)
        
        
        best_params = study.best_params
        model_path = self.model_path
        log_file_name = "best_params.json"
        log_file_path = f"{model_path}/{log_file_name}"
        
        with open(log_file_path, 'w') as f:
            json.dump(best_params, f)
        
        return best_params
        
        
    def objective(self, trial):
            
        grid_size = trial.suggest_int('grid_size', 3, 10)
        spline_order = trial.suggest_int('spline_order', 1, 4)
        range_limit = trial.suggest_int('range_limit', 3, 5)
        grid_range = [-range_limit, range_limit]
        
        lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
        
        lamb = trial.suggest_float('lamb', 0., 0.01) if self.use_reg_loss else 0. 
        mu_1 = trial.suggest_float('mu_1', 0.1, 1., log=True) if self.use_reg_loss else 1.
        mu_2 = trial.suggest_float('mu_2', 0.1, 1., log=True) if self.use_reg_loss else 1.
        
        use_orig_reg = trial.suggest_categorical("use_orig_reg", [True, False])
        
        store_acts = (use_orig_reg and lamb > 0.)
        
        self.model_config['store_acts'] = store_acts
        self.model_config['grid_size'] = grid_size
        self.model_config['spline_order'] = spline_order
        self.model_config['grid_range'] = grid_range
        
        model = NetWrapper(KanGDyn, self.model_config, self.edge_index, update_grid=False)
        model.to(torch.device(self.device))
        
        results = fit(
            model,
            self.train_data,
            self.t_train,
            self.valid_data,
            self.t_valid,
            self.test_data,
            self.t_test,
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
            save_updates=False
        )
        
        best_val_loss = min(results['validation_loss'])
        
        return best_val_loss
        
    
    def eval_model(self, best_params):
        
        self.model_config['model_path'] = f'{self.model_path}/eval'
        store_acts = (best_params['use_orig_reg'] and best_params['lamb'] > 0.)
        self.model_config['store_acts'] = store_acts
        self.model_config['grid_size'] = best_params['grid_size']
        self.model_config['spline_order'] = best_params['spline_order']
        range_limit = best_params['range_limit']
        grid_range = [-range_limit, range_limit]
        self.model_config['grid_range'] = grid_range
        
        model = NetWrapper(KanGDyn, self.model_config, self.edge_index, update_grid=False)
        model.to(torch.device(self.device))
        
        _ = fit(
            model,
            self.train_data,
            self.t_train,
            self.valid_data,
            self.t_valid,
            self.test_data,
            self.t_test,
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
            save_updates=True
        )
        
        return model
        
        