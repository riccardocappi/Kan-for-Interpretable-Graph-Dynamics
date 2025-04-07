from abc import ABC, abstractmethod
import torch
import optuna
from optuna.samplers import GridSampler
import json
import os
import logging
from train_and_eval import fit
from models.utils.NetWrapper import NetWrapper
from utils.utils import sample_from_spatio_temporal_graph
import copy
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from datasets.SpatioTemporalGraphData import SpatioTemporalGraphData


class Experiments(ABC):
    """
        Abstract class defining the experiments pipeline. To implement a specific experiment, extend this class and provide
        implementation for the two abstract methods: pre_processing, and get_model_opt.   
    """
    def __init__(self, 
                 config, 
                 n_trials,
                 model_selection_method='optuna',
                 study_name='example',
                 process_id=0,
                 store_to_sqlite = True
                 ):
        
        super().__init__()
        
        assert model_selection_method != 'optuna' or n_trials is not None
        assert model_selection_method == 'optuna' or model_selection_method == 'grid_search', 'Optimization method not supported!'
        
        self.config = config
        self.n_trials = n_trials
        self.method = model_selection_method
        
        self.device = config["device"]
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            
        self.n_iter = config['n_iter']
        
        dataset = SpatioTemporalGraphData(
            root=config['data_folder'],
            dynamics=config['dynamics'],
            t_span=config['t_span'],
            t_max=config['t_eval_steps'],
            num_samples=config['num_samples'],
            seed=config['seed'],
            n_ics=config['n_iter'],
            input_range=config['input_range'],
            device=config['device'],
            **config['integration_kwargs']
        )
        
        t_f_train = int(0.8 * len(dataset))
        
        self.training_set = dataset[:t_f_train]
        self.valid_set = dataset[t_f_train:]
        
        self.edge_index = self.training_set[0].edge_index

        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.opt = config["opt"]
        self.log = config["log"]
        
        self.process_id = process_id
        self.model_path = f'./saved_models_optuna/{config["model_name"]}/{study_name}/{str(process_id)}'
        
        self.search_space = config['search_space']      # Optuna search space
        
        # Seeds for reproducibility
        self.seed = config['seed']
        self.torch_seed = config.get("pytorch_seed", 42)    
        
        self.criterion = torch.nn.MSELoss()
        
        self.study_name = study_name
        
        logs_folder = f'{self.model_path}/optuna_logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file_path = f'{logs_folder}/optuna_logs.txt'
        
        # Forcing optuna to save logs to file
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)  # Setup the root logger.
        self.optuna_handler = logging.FileHandler(logs_file_path, mode="w")
        
        logger.addHandler(self.optuna_handler)
        optuna.logging.enable_propagation()
        
        self.current_model_state_pool = []  # List of the weights of the models trained in the current trial (every trial has multiple trainings and averages the validation losses) 
        self.current_model_arch = None      # To save the current model architecture
        
        self.best_results = {}
        self.best_params = {}   
        self.best_model = None  
        
        self.storage = "sqlite:///optuna_study.db" if store_to_sqlite else JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
        self.method = self.config.get('method', 'dopri5')
        
        
      
    def run(self):
        """
        Run the experiment pipeline. 
        """
        
        self.training_set, self.valid_set = self.pre_processing(self.training_set, self.valid_set)  # Custom pre_processing
        self.optimize() # Optuna study optimization 
        
        # Disabling optuna logger
        logging.getLogger().removeHandler(self.optuna_handler)
        optuna.logging.disable_propagation()        
        
        # Saving the best hyper-parameters found by optuna to file
        log_file_name = "best_params.json"
        log_file_path = f"{self.model_path}/{log_file_name}"
        with open(log_file_path, 'w') as f:
            json.dump(self.best_params, f)
        
        if self.best_model is not None:
            self.post_processing(self.best_model)   # Saving best_model checkpoint to file
    
    
    def optimize(self):
        """
        Optimize hyper-parameters using optuna.
        """
        if self.method == 'grid_search':
            sampler = GridSampler(self.search_space)
            n_trials = len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
            n_trials = self.n_trials
        
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{self.config["model_name"]}-{self.study_name}',
            sampler=sampler,
            storage=self.storage,
            load_if_exists=True
        )
                
        study.optimize(
            self.objective, 
            n_trials=n_trials, 
            callbacks=[self.callback], 
            catch=[AssertionError]
        )
    
    
    def callback(self, study, trial):
        """
        Function called at the end of each trial. It updates the best model found so far in the optuna study
        
        Args:
            - study : optuna study
            - trial : current optuna trial
        """
        best_params = study.best_params
        if study.best_trial == trial:
            self.best_params = best_params
            self.best_model = self.current_model_arch
            assert len(self.current_model_state_pool) > 0
            _, best_weights, best_results = min(self.current_model_state_pool, key=lambda x: x[0])    # This choice may also be random
            self.best_model.load_state_dict(best_weights)
            self.best_results = best_results
    
    
    def objective(self, trial):
        """
        Objective function optimized by optuna.
        
        Args:
            - trial : current optuna trial
        """
        R = self.config.get("R", 1)   # Number of internal training run
        
        lr_space = self.search_space.get('lr', [0.001])
        lr = trial.suggest_float('lr', lr_space[0], lr_space[-1], log=True)
        
        lamb_space = self.search_space.get('lamb', [0.])
        lamb = trial.suggest_float('lamb', lamb_space[0], lamb_space[-1], step=0.0001)
        
        batch_size_space = self.search_space.get('batch_size', [-1])
        batch_size = trial.suggest_categorical('batch_size', batch_size_space)
        
        tot_val_loss = 0.
        
        model = self.get_model_opt(trial)   # get the current model
        
        self.current_model_arch = model     # Save current model architecture
        self.current_model_state_pool.clear()
        
        for _ in range(R):
            results = fit(
                model,
                self.training_set,
                self.valid_set,
                epochs=self.epochs,
                patience=self.patience,
                lr = lr,
                lmbd=lamb,
                log=self.log,
                criterion=self.criterion,
                opt=self.opt,
                save_updates=False,
                n_iter=self.n_iter,
                batch_size=batch_size,
                method=self.method
            )
            
            best_val_loss = min(results['validation_loss'])
            tot_val_loss += best_val_loss
            self.current_model_state_pool.append(
                (best_val_loss, 
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(results) 
                )
            )    # Save model weights of each trained model
            
            model.model.reset_params()  # random initialization of model weights
                
        trial.set_user_attr("process_id", self.process_id)
        
        # return the average of validation losses over the number of runs        
        return tot_val_loss / R

    
    @abstractmethod
    def pre_processing(self, train_data, valid_data):
        """
        Custom pre-processing
        
        Args:
            - train_data : training set
            - valid_data : validation set
        """
        raise Exception('Not implemented')
    
    
    @abstractmethod
    def get_model_opt(self, trial) -> NetWrapper:
        """
        Every experiment must specify how to construct the model given an optuna trial
        
        Args:
            -trial : current optuna trial
        """
        raise Exception('Not implemented')
        
    
    def post_processing(self, best_model: NetWrapper, sample_size = -1):
        """
        Save the best model checkpoint to file.
        
        Args:
            - best_model : Best model resulting from the model selection procedure
            - sample_size : number of graph snapshot to sample from the training set (-1 samples the whole set)
        """
        # Save best results
        with open(f"{best_model.model.model_path}/results.json", "w") as f:
            json.dump(self.best_results, f)
        
        # Save best state dict
        torch.save(best_model.state_dict(), f"{best_model.model.model_path}/state_dict.pth")
        
        # Sample from the graph-time-series
        dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(
            self.training_set.raw_data_sampled[0], 
            self.edge_index, 
            sample_size=sample_size
        )
        
        # Save model checkpoint
        best_model.model.save_cached_data(dummy_x, dummy_edge_index)    