from utils.utils import load_config
from experiments.experiments_gkan import ExperimentsGKAN
from experiments.experiments_baseline import ExperimentsBaseline
import networkx as nx


def run(config_path, n_trials=10, method='optuna', model_type='GKAN'):
    config = load_config(config_path)
    G = nx.grid_2d_graph(7, 10)
    if model_type == 'GKAN':
        exp = ExperimentsGKAN(config, G, n_trials, method)
    elif model_type == 'GCN' or model_type == 'GIN':
        exp = ExperimentsBaseline(config, G, n_trials, method, model_type=model_type)
    else:
        raise Exception('Unknown model type')
    
    exp.run()
    
    
if __name__ == '__main__':
    run('./configs/config_kuramoto.yml', method='optuna', n_trials=1)
    run('./configs/config_kuramoto_gcn.yml', method='optuna', n_trials=1, model_type='GCN')
    run('./configs/config_kuramoto_gin.yml', method='optuna', n_trials=1, model_type='GIN')
    
    
    