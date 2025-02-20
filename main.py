import argparse
from utils.utils import load_config
from experiments.experiments_gkan import ExperimentsGKAN
from experiments.experiments_baseline import ExperimentsBaseline
from experiments.experiments_mpnn import ExperimentsMPNN
import networkx as nx

def run(config_path, n_trials=10, method='optuna', study_name='example', eval_model=True, process_id=0):
    config = load_config(config_path)
    model_type=config['model_type']
    # G = nx.grid_2d_graph(7, 10)
    G = nx.barabasi_albert_graph(70, 3, seed=config['seed'])
    
    if model_type == 'GKAN':
        exp = ExperimentsGKAN(config, G, n_trials, method, study_name=study_name, eval_model=eval_model, process_id=process_id)
    elif model_type in ['GCN', 'GIN']:
        exp = ExperimentsBaseline(config, G, n_trials, method, model_type=model_type, study_name=study_name, eval_model=eval_model, process_id=process_id)
    elif model_type == 'MPNN':
        exp = ExperimentsMPNN(config, G, n_trials, method, study_name=study_name, eval_model=eval_model, process_id=process_id)
    else:
        raise ValueError('Unknown model type')
    
    exp.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with different model types.')
    
    parser.add_argument('--config', default='./configs/config_kuramoto.yml', help='Path to config file')
    parser.add_argument('--method', default='optuna', help='Optimization method')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--study_name', default='example', help='Name of the optuna study to load/create')
    parser.add_argument('--process_id', type=int, default=0, help='ID for the running process')
    parser.add_argument('--eval_model', default=False, action="store_true", help='Whether to evaluate the model after the training process')
    
    
    args = parser.parse_args()
    
    run(args.config, args.n_trials, args.method, args.study_name, args.eval_model, args.process_id)
    
    
    