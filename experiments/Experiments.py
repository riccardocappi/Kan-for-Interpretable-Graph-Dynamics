from abc import ABC, abstractmethod
import torch
from utils.utils import create_datasets
from torch_geometric.utils import from_networkx
import optuna
from optuna.samplers import GridSampler
import json
import os
import logging
from train_and_eval import fit
from models.utils.NetWrapper import NetWrapper
from utils.utils import sample_from_spatio_temporal_graph


class Experiments(ABC):
    def __init__(self, 
                 config, 
                 G, 
                 n_trials,
                 model_selection_method='optuna',
                 study_name='example',
                 process_id=0):
        
        super().__init__()
        
        assert model_selection_method != 'optuna' or n_trials is not None
        assert model_selection_method == 'optuna' or model_selection_method == 'grid_search', 'Optimization method not supported!'
        
        self.config = config
        self.n_trials = n_trials
        self.method = model_selection_method
        
        self.device = config["device"]
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            
        self.t_f_train = int(0.8 * config['t_eval_steps'])
        self.n_iter = config['n_iter']
        self.training_set, self.valid_set = create_datasets(config, G, t_f_train=self.t_f_train)
        
        self.edge_index = from_networkx(G).edge_index
        self.edge_index = self.edge_index.to(torch.device(self.device))

        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        
        self.process_id = process_id
        self.model_path = f'./saved_models_optuna/{config["model_name"]}/{study_name}/{str(process_id)}'
        
        self.search_space = config['search_space']
        self.seed = config['seed']
        
        self.study_name = study_name
        
        logs_folder = f'{self.model_path}/optuna_logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file_path = f'{logs_folder}/optuna_logs.txt'
        
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)  # Setup the root logger.
        self.optuna_handler = logging.FileHandler(logs_file_path, mode="w")
        
        logger.addHandler(self.optuna_handler)
        optuna.logging.enable_propagation()
        
        self.current_model = None
        self.current_results = {}
        
        self.best_params = {}
        self.best_model = None
        self.best_results = {}
        
      
    def run(self):
        self.training_set, self.valid_set = self.pre_processing(self.training_set, self.valid_set)
        self.optimize()
        
        logging.getLogger().removeHandler(self.optuna_handler)
        optuna.logging.disable_propagation()        
        
        log_file_name = "best_params.json"
        log_file_path = f"{self.model_path}/{log_file_name}"
        with open(log_file_path, 'w') as f:
            json.dump(self.best_params, f)
        
        torch.save(self.best_model.state_dict(), f'{self.model_path}/best_state_dict.pth')
        
        with open(f"{self.model_path}/results.json", "w") as outfile: 
            json.dump(self.best_results, outfile)
        
    
        self.post_processing(self.best_model)
    
    
    def optimize(self):
        if self.method == 'grid_search':
            sampler = GridSampler(self.search_space)
            n_trials = len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler()
            n_trials = self.n_trials
        
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{self.config["model_name"]}-{self.study_name}',
            sampler=sampler,
            storage="sqlite:///optuna_study.db",
            load_if_exists=True
        )
                
        study.optimize(self.objective, n_trials=n_trials, callbacks=[self.callback], catch=[AssertionError])
    
    
    def callback(self, study, trial):
        best_params = study.best_params
        if study.best_trial == trial:
            self.best_params = best_params
            self.best_model = self.current_model
            self.best_results = self.current_results
    
    
    def objective(self, trial):
        self.current_model = self.get_model_opt(trial)
        
        lr_space = self.search_space.get('lr', [0.001])
        lr = trial.suggest_float('lr', lr_space[0], lr_space[-1])
        
        lamb_space = self.search_space.get('lamb', [0.])
        lamb = trial.suggest_float('lamb', lamb_space[0], lamb_space[-1])
        
        batch_size_space = self.search_space.get('batch_size', [-1])
        batch_size = trial.suggest_categorical('batch_size', batch_size_space)
        
        stride_space = self.search_space.get('stride', [1])
        stride = trial.suggest_categorical('stride', stride_space)
        
        self.current_results = fit(
            self.current_model,
            self.training_set,
            self.valid_set,
            epochs=self.epochs,
            patience=self.patience,
            lr = lr,
            lmbd=lamb,
            log=self.log,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            save_updates=False,
            n_iter=self.n_iter,
            batch_size=batch_size,
            t_f_train=self.t_f_train,
            stride=stride
        )
        
        best_val_loss = min(self.current_results['validation_loss'])
        
        trial.set_user_attr("process_id", self.process_id)
        
        return best_val_loss
    
    
    @abstractmethod
    def pre_processing(self, train_data, valid_data):
        raise Exception('Not implemented')
    
    
    @abstractmethod
    def get_model_opt(self, trial) -> NetWrapper:
        raise Exception('Not implemented')
        
    
    def post_processing(self, best_model: NetWrapper, sample_size = -1):
        
        dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(self.training_set.data[0], 
                                                                    self.edge_index, 
                                                                    sample_size=sample_size)
        best_model.model.save_cached_data(dummy_x, dummy_edge_index)
        
        