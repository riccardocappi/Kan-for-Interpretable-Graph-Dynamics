from abc import ABC, abstractmethod
import torch
from utils.utils import create_datasets
from torch_geometric.utils import from_networkx
import optuna
from optuna.samplers import GridSampler
import json
import os
import logging

class Experiments(ABC):
    def __init__(self, 
                 config, 
                 G, 
                 n_trials,
                 search_space = None,
                 model_selection_method='optuna',
                 t_f_train=240):
        
        super().__init__()
        
        assert model_selection_method != 'optuna' or n_trials is not None
        assert model_selection_method == 'optuna' or model_selection_method == 'grid_search', 'Optimization method not supported!'
        assert model_selection_method != 'grid_search' or search_space is not None
        
        self.config = config
        self.n_trials = n_trials
        self.method = model_selection_method
        
        self.device = config["device"]
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            
        self.t_f_train = t_f_train
        self.n_iter = config['n_iter']
        self.training_set, self.valid_set = create_datasets(config, G, t_f_train=self.t_f_train)
        
        self.edge_index = from_networkx(G).edge_index
        self.edge_index = self.edge_index.to(torch.device(self.device))

        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        
        self.use_reg_loss = config['reg_loss']
        self.model_path = f'./saved_models_optuna/{config["model_name"]}'
        self.search_space = search_space
        
        logs_folder = f'{self.model_path}/optuna_logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file_path = f'{logs_folder}/optuna_logs.txt'
        
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)  # Setup the root logger.
        self.optuna_handler = logging.FileHandler(logs_file_path, mode="w")
        
        logger.addHandler(self.optuna_handler)
        optuna.logging.enable_propagation()
    
    
    def run(self):
        self.training_set, self.valid_set = self.pre_processing(self.training_set, self.valid_set)
        best_params = self.optimize()
        
        logging.getLogger().removeHandler(self.optuna_handler)
        optuna.logging.disable_propagation()
        
        self.post_processing(best_params)
    
    
    @abstractmethod
    def pre_processing(self, train_data, valid_data):
        raise Exception('Not implemented')
    
    
    def optimize(self):
        if self.method == 'grid_search':
            sampler = GridSampler(self.search_space)
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
    
    
    @abstractmethod
    def objective(self, trial):
        raise Exception('Not implemented')
    
    
    @abstractmethod
    def post_processing(self, best_params):
        raise Exception('Not implemented')
    
    