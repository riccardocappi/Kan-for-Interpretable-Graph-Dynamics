import torch
import numpy as np
from .utils import get_conf, create_datasets
from train_and_eval import fit
import optuna
from optuna.samplers import GridSampler
import json
from models.KanGDyn import KanGDyn



class ModelSelector():
    
    def __init__(self, config, G, noise_level=None, n_trials=None, method='grid_search'):
        
        super().__init__()
        
        assert method != 'optuna' or n_trials is not None
        assert method == 'optuna' or method == 'grid_search', 'Optimization method not supported!'
        
        self.n_trials = n_trials
        self.method = method
        rng = np.random.default_rng(seed=config.get("seed", 42))
        name_suffix = f"_noise_{noise_level}" if noise_level is not None else ''
        
        root = config["root"]
        name = config["name"] + name_suffix
        self.device = config["device"]
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            
        conf = get_conf(config, noise_level, rng)
        self.train_dataset, self.valid_dataset, self.test_dataset = create_datasets(conf, root=root, name=name, graph=G)
        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        self.input_noise = config["add_noise_input"]
        self.model_path = f'./saved_models_optuna/{config["model_name"]}' + name_suffix
        self.use_reg_loss = config['reg_loss']
        
        # Model hyper-params
        self.g_hidden_layers = config['g_hidden_layers']
        self.h_hidden_layers = config['h_hidden_layers']
        self.input_range = [config["vmin"], config["vmax"]]
        self.epsilon = config["step_size"]
        
        
    
    def optimize(self):
        # Maybe get search_space from config
        if self.method == 'grid_search':
            search_space = {
                'grid_size': [5],
                'spline_order': [3],
                'lr': [0.01, 0.005],
                'batch': [32],
                'lamb': [0.0001, 0.001] if self.use_reg_loss else [0.],
                'mu_1': [0.5, 1.] if self.use_reg_loss else [1.],
                'mu_2': [0.5, 1.] if self.use_reg_loss else [1.],
                'norm': [False]
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
        
        lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
        batch = trial.suggest_categorical('batch', [32, 64])
        
        lamb = trial.suggest_float('lamb', 0.0001, 0.01, log=True) if self.use_reg_loss else 0. 
        mu_1 = trial.suggest_float('mu_1', 0.1, 1., log=True) if self.use_reg_loss else 1.
        mu_2 = trial.suggest_float('mu_2', 0.1, 1., log=True) if self.use_reg_loss else 1.
        
        norm = trial.suggest_categorical("norm", [True, False])
        
        model = KanGDyn(
            h_hidden_layers=self.h_hidden_layers,
            g_hidden_layers=self.g_hidden_layers,
            grid_range=[self.input_range[0], self.input_range[1]],
            model_path=self.model_path,
            epsilon=self.epsilon,
            device=self.device,
            store_acts=False,
            norm=norm,
            grid_size=grid_size,
            spline_order=spline_order
        )
        
        results = fit(model, self.train_dataset, self.valid_dataset, self.test_dataset, batch_size=batch, epochs=self.epochs, 
                      criterion=torch.nn.L1Loss(), lr=lr, opt = self.opt, log=self.log, patience=self.patience, lamb=lamb, 
                      use_orig_reg=False, save_updates=False, mu_1=mu_1, mu_2=mu_2, update_grid=False)
        
        best_val_loss = min(results['validation_loss'])
        
        return best_val_loss
        
    
    def eval_model(self, best_params):
        
        model = KanGDyn(
            self.h_hidden_layers,
            self.g_hidden_layers,
            grid_range=[self.input_range[0], self.input_range[1]],
            model_path=f'{self.model_path}/eval',
            epsilon=self.epsilon,
            device=self.device,
            store_acts=False,
            norm=best_params['norm'],
            grid_size=best_params['grid_size'],
            spline_order=best_params['spline_order']
        )
        
        _ = fit(model, self.train_dataset, self.valid_dataset, self.test_dataset, batch_size=best_params['batch'], epochs=self.epochs, 
                criterion=torch.nn.L1Loss(), lr=best_params['lr'], opt = self.opt, log=self.log, patience=self.patience, 
                lamb=best_params.get('lamb', 0.), use_orig_reg=False, save_updates=True, mu_1=best_params.get('mu_1', 1.), 
                mu_2=best_params.get('mu_2', 1.), update_grid=False)
        
        return model
        
        