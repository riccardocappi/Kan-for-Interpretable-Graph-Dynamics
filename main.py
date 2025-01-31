import argparse
from utils.utils import load_config
from experiments.experiments_gkan import ExperimentsGKAN
from experiments.experiments_baseline import ExperimentsBaseline
import networkx as nx

def run(config_path, n_trials=10, method='optuna'):
    config = load_config(config_path)
    model_type=config['model_type']
    G = nx.grid_2d_graph(7, 10)
    
    if model_type == 'GKAN':
        exp = ExperimentsGKAN(config, G, n_trials, method)
    elif model_type in ['GCN', 'GIN']:
        exp = ExperimentsBaseline(config, G, n_trials, method, model_type=model_type)
    else:
        raise ValueError('Unknown model type')
    
    exp.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with different model types.')
    
    parser.add_argument('--config', default='./configs/config_kuramoto.yml', help='Path to config file')
    parser.add_argument('--method', default='optuna', help='Optimization method')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    
    args = parser.parse_args()
    
    run(args.config, args.n_trials, args.method)
    
    
    