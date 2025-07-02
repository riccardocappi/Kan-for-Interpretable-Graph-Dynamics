from abc import ABC, abstractmethod
import torch
import optuna
from optuna.samplers import GridSampler
import json
import os
import logging
from train_and_eval import fit
from models.utils.ODEBlock import ODEBlock
from utils.utils import sample_from_spatio_temporal_graph
import copy
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from datasets.SyntheticData import SyntheticData
from datasets.data_utils import dynamics_name
from utils.utils import SCORES
import yaml
from tsl.data.preprocessing.scalers import MinMaxScaler
from datasets.SpatioTemporalGraph import SpatioTemporalGraph
from train_and_eval import eval_model


class Experiments(ABC):
    """
        Abstract class defining the experiments pipeline. To implement a specific experiment, extend this class and provide
        implementation for the abstract method: get_model_opt.   
    """
    def __init__(self, 
                 config, 
                 n_trials,
                 model_selection_method='optuna',
                 study_name='example',
                 process_id=0,
                 snr_db = -1
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
        self.horizon = config.get('horizon', 1)
        self.history = config.get('history', 1) 
        self.preprocess_data = config.get('preprocess_data', False)
        self.predict_deriv = config.get("predict_deriv", False)
        self.snr_db = snr_db
        
        if config['name'] in dynamics_name:
            dataset = SyntheticData(
                root=config['data_folder'],
                dynamics=config['name'],
                t_span=config['t_span'],
                t_max=config['t_eval_steps'],
                num_samples=config['num_samples'],
                seed=config['seed'],
                n_ics=config['n_iter'],
                input_range=config['input_range'],
                device=self.device,
                horizon = self.horizon,
                history = self.history,
                stride=config.get('stride', 5),
                predict_deriv=self.predict_deriv,
                snr_db=self.snr_db,
                **config['integration_kwargs']
            )
        else:
            raise NotImplementedError()
        
        total_len = len(dataset)

        # Compute split sizes
        train_end = int(0.8 * total_len)

        # Create splits
        self.training_set = dataset[:train_end]
        self.valid_set = dataset[train_end:]
    
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
        
        self.criterion = SCORES[config.get('criterion', 'MSE')]
        
        self.study_name = study_name
        
        logs_folder = f'{self.model_path}/optuna_logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file_path = f'{logs_folder}/optuna_logs.txt'
        
        # Forcing optuna to save logs to file
        logger = logging.getLogger()

        logger.setLevel(logging.INFO)  # Setup the root logger.
        self.optuna_handler = logging.FileHandler(logs_file_path, mode="w")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.optuna_handler.setFormatter(formatter)
        
        logger.addHandler(self.optuna_handler)
        optuna.logging.enable_propagation()
        
        self.current_model_state_pool = []  # List of the weights of the models trained in the current trial (every trial has multiple trainings and averages the validation losses) 
        self.current_model_arch = None      # To save the current model architecture
        
        self.best_results = {}
        self.best_params = {}   
        self.best_model = None 

        storage = self.config.get('storage', 'journal')
        
        if storage == 'sqlite':
            store_to_sqlite = True
        elif storage == 'journal':
            store_to_sqlite = False
        else:
            raise ValueError("Not supported storage backend!") 
        
        self.storage = "sqlite:///optuna_study.db" if store_to_sqlite else JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
        self.integration_method = self.config.get('method', 'dopri5')
        
        self.scaler = None
        self.save_cache_data = self.config.get('save_cache_data', True)
        
        # Save a copy of config file to study's folder
        copy_config_path = f'./saved_models_optuna/{config["model_name"]}/{study_name}/config.yml'
        if not os.path.exists(copy_config_path):
            with open(copy_config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
         
    def run(self):
        """
        Run the experiment pipeline. 
        """
        self.scaler = self.pre_processing(self.training_set) if self.preprocess_data else None
        
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
            sampler = optuna.samplers.TPESampler(seed=self.seed)
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
            self._save_ckpt(self.best_model)
            

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
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trial {trial.number}: num params: {total_params}")
        
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
                batch_size=batch_size,
                scaler = self.scaler,
                pred_deriv=self.predict_deriv
            )
            
            best_val_loss = min(results['validation_loss'])
            tot_val_loss += best_val_loss
            self.current_model_state_pool.append(
                (best_val_loss, 
                copy.deepcopy(model.state_dict()),
                copy.deepcopy(results) 
                )
            )    # Save model weights of each trained model
            
            model.reset_params()  # random initialization of model weights
                
        trial.set_user_attr("process_id", self.process_id)
        
        # return the average of validation losses over the number of runs        
        return tot_val_loss / R
    
    
    def pre_processing(self, training_set:SpatioTemporalGraph):
        all_train_x = torch.cat([data.x.view(-1) for data in training_set], dim=0)  
        scaler = MinMaxScaler(out_range=(-1, 1))
        scaler.fit(all_train_x.detach().cpu())
        
        scaler.scale = scaler.scale.to(torch.device(self.device))
        scaler.bias = scaler.bias.to(torch.device(self.device))
        
        return scaler
    
    
    @abstractmethod
    def get_model_opt(self, trial) -> ODEBlock:
        """
        Every experiment must specify how to construct the model given an optuna trial
        
        Args:
            -trial : current optuna trial
        """
        raise NotImplementedError()
        
    
    def post_processing(self, best_model: ODEBlock, sample_size = -1, raw_data=None):
        """
        Save the best model checkpoint to file.
        
        Args:
            - best_model : Best model resulting from the model selection procedure
            - sample_size : number of graph snapshot to sample from the training set (-1 samples the whole set)
            - raw_data : raw data to sample from (if None, it will be sampled from the training set)
        """
        # Save best results
        self._save_ckpt(best_model)
        
        if self.save_cache_data:
            if raw_data is None:
                if self.scaler is not None:
                    raw_data = self.scaler.transform(self.training_set.raw_data_sampled[0])
                else:
                    raw_data = self.training_set.raw_data_sampled[0]
            
            t = self.training_set.t_sampled[0] if self.config.get("include_time", False) else None
            
            dummy_x, dummy_edge_index, dummy_t, dummy_edge_attrs = sample_from_spatio_temporal_graph(
                raw_data, 
                self.training_set[0].edge_index,
                edge_attr=self.training_set[0].edge_attr,
                t=t,
                sample_size=sample_size
            )
            
            # Save model checkpoint
            best_model.save_cached_data(dummy_x, dummy_edge_index, dummy_t=dummy_t, dummy_edge_attr = dummy_edge_attrs)
        
    
    def _save_ckpt(self, best_model:ODEBlock):
        with open(f"{best_model.model_path}/results.json", "w") as f:
            json.dump(self.best_results, f)
        
        # Save best state dict
        torch.save(best_model.state_dict(), f"{best_model.model_path}/state_dict.pth")