import torch
import numpy as np
from .utils import get_conf, create_datasets
from train_and_eval import train_and_eval
import optuna
from optuna.samplers import GridSampler
import json


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
        self.model_path = f'./saved_models_optuna/{config["model_name"]}' + name_suffix
        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        self.input_noise = config["add_noise_input"]
        self.hidden_layers = config['hidden_layers']
        
        self.input_range = [config["vmin"], config["vmax"]]
        self.epsilon = config["step_size"]
        
        
    
    def optimize(self):
        # Maybe get search_space from config
        if self.method == 'grid_search':
            search_space = {
                'grid_size': [5],
                'spline_order': [3],
                'lr': [0.001, 0.005, 0.01],
                'batch': [32],
                'lamb': [0.0001, 0.001, 0.01] if self.input_noise else [0.],
                'mu_1': [0.1, 0.5, 1.] if self.input_noise else [1.],
                'mu_2': [0.1, 0.5, 1.] if self.input_noise else [1.],
                'aggr_first': [True, False],
                'norm': [True, False]
            }
            sampler = GridSampler(search_space)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(self.objective, n_trials=len(sampler._all_grids))
        
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=self.n_trials)
        
        return study.best_params
        
        
    def objective(self, trial):
        
        # TODO: Handle use_orig_reg param
             
        grid_size = trial.suggest_int('grid_size', 3, 10)
        spline_order = trial.suggest_int('spline_order', 1, 4)
        
        lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
        batch = trial.suggest_categorical('batch', [32, 64])
        
        lamb = trial.suggest_float('lamb', 0.0001, 0.01, log=True) if self.input_noise else 0. 
        mu_1 = trial.suggest_float('mu_1', 0.1, 1., log=True) if self.input_noise else 1.
        mu_2 = trial.suggest_float('mu_2', 0.1, 1., log=True) if self.input_noise else 1.
        
        aggregate_first = trial.suggest_categorical("aggr_first", [True, False])
        norm = trial.suggest_categorical("norm", [True, False])
        
        
        best_val_loss = train_and_eval(hidden_layers=self.hidden_layers,
                            model_path=self.model_path,
                            aggregate_first=aggregate_first,
                            input_range=self.input_range,
                            epsilon=self.epsilon,
                            device=self.device,
                            gconv_norm='A_hat_norm',
                            norm=norm,
                            aggr='add',
                            grid_size=grid_size,
                            spline_order=spline_order,
                            train_dataset=self.train_dataset,
                            valid_dataset=self.valid_dataset,
                            test_dataset=self.test_dataset,
                            batch=batch,
                            epochs=self.epochs,
                            lr=lr,
                            opt=self.opt,
                            log=self.log,
                            patience=self.patience,
                            lamb=lamb,
                            mu_1=mu_1,
                            mu_2=mu_2,
                            eval_model=False,
                            store_acts=self.input_noise,
                            update_grid=False)
        
        return best_val_loss
        
    
    def eval_model(self, best_params):
        model_path = self.model_path
        log_file_name = "best_params.json"
        log_file_path = f"{model_path}/{log_file_name}"
        
        with open(log_file_path, 'w') as f:
            json.dump(best_params, f)
        
        # Evaluate the best resulting model
        _ = train_and_eval(hidden_layers=self.hidden_layers,
                model_path=f"{model_path}/eval",
                aggregate_first=best_params['aggr_first'], 
                input_range=self.input_range,
                aggr='add',
                norm=best_params['norm'],
                epsilon=self.epsilon,
                device=self.device,
                gconv_norm='A_hat_norm',
                grid_size=best_params.get('grid_size',5),
                spline_order=best_params.get('spline_order', 3),
                train_dataset=self.train_dataset,
                valid_dataset=self.valid_dataset,
                test_dataset=self.test_dataset,
                batch=best_params['batch'],
                epochs=self.epochs,
                lr=best_params['lr'],
                opt=self.opt,
                log=self.log,
                patience=self.patience,
                lamb=best_params.get('lamb', 0.),
                mu_1=best_params.get('mu_1', 1.),
                mu_2=best_params.get('mu_2', 1.),
                eval_model=True,
                update_grid=False)
        