import argparse
from utils.utils import load_config
from experiments.experiments_gkan import ExperimentsGKAN
from experiments.experiments_mpnn import ExperimentsMPNN
import torch


def set_pytorch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config_path, n_trials=10, method='optuna', study_name='example', process_id=0):
    config = load_config(config_path)   # Load yml config file 
    
    set_pytorch_seed(seed=config["pytorch_seed"])   # Set seed
    
    model_type=config['model_type']
    pred_deriv = config.get('predict_deriv', False)
    n_trials = 10 if not pred_deriv else n_trials
           
    if model_type == 'GKAN':
        exp = ExperimentsGKAN(config, n_trials, method, study_name=study_name, process_id=process_id)
    elif model_type == 'MPNN':
        exp = ExperimentsMPNN(config, n_trials, method, study_name=study_name, process_id=process_id)
    else:
        raise ValueError('Unknown model type')
    
    exp.run()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run experiments with different model types.') 
    parser.add_argument('--config', default='./configs/config_kuramoto.yml', help='Path to config file')
    parser.add_argument('--method', default='optuna', help='Optimization method. can be optuna or grid_search')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of optuna trials')
    parser.add_argument('--study_name', default='example', help='Name of the optuna study to load/create')
    parser.add_argument('--process_id', type=int, default=0, help='ID for the running process')
    
    
    args = parser.parse_args()
    
    run(args.config, args.n_trials, args.method, args.study_name, args.process_id)
    
    
    