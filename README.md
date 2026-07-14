# Discovering Generalizable Governing Equations for Graph Dynamical Systems with Interpretable Neural Networks

This repository contains the code used to train and evaluate Kolmogorov-Arnold Network (KAN) based models for learning the governing equations of graph dynamical systems, and to recover interpretable, symbolic closed-form expressions from the trained models.

## Table of contents
- [Installation](#installation)
- [Project structure](#project-structure)
- [Running experiments](#running-experiments)
- [Config specification](#config-specification)
- [Results](#results)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/riccardocappi/Kan-for-Interpretable-Graph-Dynamics.git
cd Kan-for-Interpretable-Graph-Dynamics
```

### 2. Install the dependencies

Create a conda environment with Python 3.12 and install the pinned dependencies from `requirements.txt` with pip:

```bash
conda create -n myenv python=3.12.0
conda activate myenv

pip install -r requirements.txt
```

`requirements.txt` pins PyTorch 2.3.1 with CUDA 11.8 support, PyTorch Geometric and its companion packages (`torch-scatter`, `torch-sparse`), and the remaining Python dependencies (`optuna`, `torchdiffeq`, `pysr`, `tsl`, `sympytorch`, etc.), matching the environment the repository was developed and tested with. If you need a different PyTorch/CUDA combination (or a CPU-only install), edit the `--extra-index-url`/`--find-links` lines and the `torch`/`torch_scatter`/`torch_sparse` versions at the top of `requirements.txt` accordingly — see the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG wheel](https://data.pyg.org/whl/) install pages.

## Project structure

```
.
├── main.py                  # Entry point: parses CLI args and launches an experiment
├── requirements.txt           # Pinned Python dependencies (install via pip)
├── configs/                   # YAML configuration files for each experiment/dynamics
├── experiments/                # Experiment pipelines (pre-processing, model selection, checkpointing)
│   ├── Experiments.py            # Abstract base class defining the experiment pipeline
│   ├── experiments_gkan.py        # Pipeline for GKAN-ODE models
│   ├── experiments_mpnn.py        # Pipeline for MPNN-ODE models
│   ├── experiments_llc.py         # Pipeline for LLC models
│   └── experiments_mlp.py         # Pipeline for MLP-ODE models
├── models/                    # Model definitions
│   ├── GKAN_ODE.py               # GKAN-ODE model
│   ├── kan/                      # KAN and KAN-layer implementations
│   ├── baseline/                  # Baseline models (MPNN, LLC, MLP)
│   └── utils/                     # Shared building blocks (MPNN layer, MLP, ODEBlock)
├── datasets/                  # Synthetic and real-world dataset generation utilities
│   ├── SyntheticData.py           # Generates synthetic trajectories via scipy's solve_ivp
│   ├── RealEpidemics.py           # Loaders for real-world epidemic datasets
│   ├── SpatioTemporalGraph.py     # Graph/spatio-temporal data structures
│   └── data_utils.py              # Library of governing-equation/dynamics functions
├── utils/                     # General utilities (config loading, curve fitting, ...)
├── train_and_eval.py           # Training loop, evaluation and early stopping logic
├── post_processing*.py         # Post-processing scripts (symbolic regression, metrics, plots)
```

## Running experiments

Experiments are orchestrated by the `Experiments` abstract class, which implements a three-stage pipeline:
1. **Pre-processing** – generates (or loads) the dataset for the selected dynamics.
2. **Model selection** – searches the hyper-parameter space with [Optuna](https://optuna.org/).
3. **Checkpointing** – saves the best model found, along with any cached data needed for downstream symbolic regression.

Each concrete subclass of `Experiments` (e.g. `ExperimentsGKAN`, `ExperimentsMPNN`, `ExperimentsLLC`, `ExperimentsMLP`) only needs to specify:
- how to pre-process the data, and
- how to build the model for a given Optuna `trial`.

For example, `ExperimentsGKAN.get_model_opt` builds a GKAN-ODE model from the parameters sampled by the current trial:

```python
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
        conv=conv,
        model_path=f"{self.model_path}/gkan",
        adjoint=self.config.get('adjoint', False),
        integration_method=self.integration_method,
        lmbd_g=lamb_g,
        lmbd_h=lamb_h,
        atol=self.config.get('atol', 1e-6),
        rtol=self.config.get('rtol', 1e-3),
        predict_deriv=self.predict_deriv
    )

    return model.to(torch.device(self.device))
```

The returned model must be an instance of `ODEBlock`, which integrates the underlying message-passing neural network with `torchdiffeq` and defines the interface expected by the `Experiments` pipeline. Concretely, every model must implement:
- **`regularization_loss`** – the regularization term added to the training loss (e.g. the L1 norm of the model's weights; `0.` for non-KAN-based models). Called from `fit` in `train_and_eval.py`.
- **`save_cached_data`** – called during post-processing to persist the model's inputs/outputs for later symbolic regression.
- **`reset_params`** – resets the model's weights; called before each new run inside `Experiments.objective`.

Datasets are generated with `scipy`'s `solve_ivp` integrator. The available governing equations/dynamics are listed in [datasets/data_utils.py](datasets/data_utils.py), and dataset generation itself is implemented in the `SyntheticData` class.

### Command-line arguments

| Argument | Description | Default |
|---|---|---|
| `--config` | Path to the `.yml` file describing the experiment | `./configs/config_kuramoto.yml` |
| `--method` | Hyper-parameter search method: `optuna` or `grid_search`. With `optuna`, sampling uses the `TPESampler` | `optuna` |
| `--n_trials` | Number of Optuna trials. Ignored when `--method=grid_search`, in which case every combination in the hyper-parameter grid is run | `10` |
| `--study_name` | Name of the Optuna study | `example` |
| `--process_id` | Id of the current process. Each process writes its logs to its own folder, so parallel processes sharing a study must use distinct ids | `0` |

### Example

```bash
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=0
```

To run several parallel processes on the same study, launch `main.py` multiple times with different `--process_id` values:

```bash
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=0
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto --process_id=1
```

These processes share the same Optuna study, backed by a Optuna storage system. The full study name is the concatenation of the `model_name` field in the config file and the `--study_name` CLI argument. For instance, `model_name: 'model-kuramoto-gkan'` together with `--study_name=kuramoto` produces the study name `model-kuramoto-gkan-kuramoto`. Two processes therefore share a study only if both `model_name` and `--study_name` match.

Running an experiment creates the following output structure:

```
saved_models_optuna/
└── model_name/
    └── study_name/
        └── process_id/
            └── optuna_logs/
```

## Config specification

The `.yml` config file passed via `--config` fully describes an experiment and is organized into two sections.

### General experiment arguments

| Key | Description |
|---|---|
| `name` | Name of the dynamics |
| `model_name` | Name of the model |
| `model_type` | `"MPNN"`, `"GKAN"`, `"LLC"`, or `"MLP"` |
| `epochs` | Number of training epochs |
| `patience` | Patience for early stopping |
| `opt` | Optimizer: `"Adam"` or `"LBFGS"` |
| `log` | Frequency (in epochs) at which logs are saved to file |
| `t_span` | Time span of the numerical integrator used to generate the datasets |
| `t_eval_steps` | Number of generated time steps in the dataset |
| `seed` | Seed for data generation |
| `pytorch_seed` | Seed for PyTorch |
| `device` | `"cpu"` or `"cuda"` |
| `input_range` | Input range for node features |
| `in_dim` | Dimensionality of the input feature matrix |
| `n_iter` | Number of initial conditions |
| `integration_kwargs` | Additional keyword arguments passed to the dynamics function |
| `R` | Number of training runs per hyper-parameter combination |
| `atol` | Absolute tolerance of the numerical integrator |
| `rtol` | Relative tolerance of the numerical integrator |
| `adjoint` | Whether to use the adjoint sensitivity method |
| `include_time` | If `True`, includes time as an input feature |
| `preprocess_data` | If `True`, applies preprocessing steps |
| `stride` | Stride of the sliding window |
| `storage` | Storage backend/location for experiment results |
| `save_cache_data` | Whether to save the final results needed for symbolic regression |
| `data_folder` | Path to the folder where the dataset is stored/generated |
| `criterion` | Training loss: `"MAE"` or `"MSE"` |
| `method` | ODE solver integration method (e.g. `"dopri5"`, a Runge-Kutta method) |

### Hyper-parameter search space

The second section defines the hyper-parameter search space, accessible at runtime via `self.search_space` on the `Experiments` instance. The mandatory hyper-parameters are:
- `lr` – learning rate
- `batch_size` – batch size

## Results

> **Note:** the post-processing scripts/notebooks in this repository will **not** run out of the box, since they expect the saved outputs of a full experiment run, which were too large to include in this submission. To reproduce the plots and post-processed results, run the experiments from scratch and update the relevant file paths in the notebooks and Python scripts accordingly.
