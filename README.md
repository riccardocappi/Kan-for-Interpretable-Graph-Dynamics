# GKAN-ODE

## Code organization
The pipeline logic is described by the `Experiments` class and it includes three components:
- Pre-processing.
- Model selection with `optuna`.
- Saving best model checkpoint to file

`Experiments` is an abstract class and defines some abstract methods that each specific experiment class must implement. In particular, each sub-class of `Experiments` must specify how to pre process data and how to construct the current model, given an `optuna` trial. For example, the `ExperimentsGKAN` class defines the experiments pipeline for the GKAN-ODE models. It implements the `get_model_opt` abstract method by constructing a GKAN model with the parameters returned by the current optuna trial.
```
def get_model_opt(self, trial):
    ...
    g_net = KAN(**g_net_config)
    h_net = KAN(**h_net_config)        
    
    conv = MPNN(
        h_net=h_net,
        g_net=g_net,
        message_passing=self.config.get("message_passing", True),
        include_time=self.config.get("include_time", True)
    )
    
    model = GKAN_ODE(
        conv = conv,
        model_path = f"{self.model_path}/gkan",
        adjoint=self.config.get('adjoint', False),
        integration_method=self.integration_method,
        lmbd_g=lamb_g,
        lmbd_h=lamb_h,
        atol=self.config.get('atol', 1e-6), 
        rtol=self.config.get('rtol', 1e-3),
        predict_deriv=self.predict_deriv
    )

    model = model.to(torch.device(self.device))
        
    return model


```
Note that the returned model must be instance of `ODEBlock`. This class integrates its Message Passing Neural Network using `torchdiffeq`. 

`ODEBlock` provides the general structure that each implemented model must follow in order to properly work with the Experiments pipeline. In particular, each model must implement the following three methods:
- regularization_loss: Computes the regularization loss (e.g. L1 norm of model's weights. Can be also 0. for non-KAN-based models). This function is called during the training process in the `fit` method defined in `train_and_eval.py`.

- save_cached_data: This function is called in the post-processing step of `Experiments`, when saving model checkpoint. Here you should save to file model's outputs and inputs that can be used later for symbolic regression. 

- reset_params: reset the parameters of the model. This function is called to reset model's weights after each run in the `objective` function of the `Experiments` class.

To generate datasets, we use the `scipy` numerical integrator `solve_ivp`. The list of considered dynamics can be found in the `data_utils.py` file. The code to generate the datasets can be found instead in the `SyntheticData` class.

## Usage
An experiment can be run with different arguments:
- `--config`: The path to the .yml file that specifies the configuration of the experiment

- `--method`: The optuna searching method. It can be "optuna" or "grid_search". If "optuna" is selected, the `TPESampler` is used to sample from the hyper-parameters space. Default is "optuna"

- `--n_trials`: Number of optuna trials. Note that if grid_search method is specified, this argument will be ignored, and will be executed as many trials as the number of combinations of the hyper-parameter grid.

- `--study_name`: Name of the optuna study

- `--process_id`: Id of the current process. Each process id has its own folder in which optuna logs are saved. Each parallel process on the same study should have a unique process_id.

Examples:
```
python main.py --config=./configs/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=0
```
To run multiple parallel processes is sufficient to execute the main file multiple times, with different process_id:

```
python main.py --config=./configs/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=0
python main.py --config=./configs/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=1
```

These processes share the same optuna study, which is stored inside a SQLite DB named "sqlite:///optuna_study.db". The actual name of each optuna study is composed of two strings: The `model_name` argument specified in the config.yml file, and the `--study_name` argument. For example, if I specify `model_name: 'model-kuramoto-gkan'` and `--study_name = kuramoto`, the final study name will be 'model-kuramoto-gkan-kuramoto'. Therefore, two processes share the same optuna study if they specify the same `model_name` and the same `--study_name`.

When an Experiment is run, it creates a folder with the following structure:

```
___ saved_models_optuna
    |___ model_name
        |___ study_name
            |___ process_id
                |___ optuna_logs
```

## Config specification
The config.yml file should contain the specifics of the experiment. It is divided in two parts.
### General arguments for setting up the experiment
The first set of arguments that must be present in the config file is the following:
- `name`: Name of the dynamics
- `model_name`: Name of the model
- `model_type`: It can be "MPNN" or "GKAN"
- `epochs`: Number of epochs
- `patience`: Patience for early stopping
- `opt`: Optimizer. It can be "Adam" or "LBFGS"
- `log`: How often to save logs to file
- `t_span`: Time span of the numerical integrator when generating the datasets
- `t_eval_steps`: Number of generated time steps of the dataset
- `seed`: Seed for replicate data generation
- `pytorch_seed`: Seed for pytorch
- `device`: Device. It can be "cpu" or "cuda"
- `input_range`: Input range for node features
- `in_dim`: Dimensionality of input feature matrix
- `n_iter`: Number of initial conditions
- `integration_kwargs`: Any additional argument to pass to the specified dynamics function.
- `R`: Number of training runs for each combination of hyper-parameters.
- `atol`:   
  Absolute tolerance used by the numerical integrator

- `rtol`: 
  Relative tolerance for the numerical integrator.

- `adjoint`:
  Whether to use the adjoint sensitivity method

- `include_time`: 
  If `True`, includes the time variable as part of the input features

- `horizon`: 
  Number of future steps the model attempts to predict during training.

- `history`: 
  Number of past time steps provided as input to the model.

- `preprocess_data`: 
  If `True`, applies preprocessing steps

- `stride`: stride of sliding windows

- `storage`:  
  Specifies the storage type or location for experiment results.

- `save_cache_data`:
  Whether to save final results for symbolic regression


- `data_folder`: 
  Path to the folder where the dataset is stored or will be saved after generation.

- `criterion`: *(str)*  
  Loss function used for training. Common options include `"MAE"` (Mean Absolute Error) or `"MSE"` (Mean Squared Error).

- `method`: *(str)*  
  ODE integration method used by the solver. `"dopri5"` is a Runge-Kutta method.


### Hyper-parameter search space
The search space of the hyper-parameters. Mandatory hyper-parameters are:
- `lr`: Learning rate
- `batch_size`: Size of the sliding window
- `stride`: Stride of the sliding windows

The search space dictionary can be accessed in the Experiment class with `self.search_space`



