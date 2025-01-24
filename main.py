from utils.utils import load_config
from experiments.experiments_gkan import ExperimentsGKAN
import networkx as nx


def run(config_path, n_trials=10, method='optuna'):
    config = load_config(config_path)
    G = nx.grid_2d_graph(7, 10)
    exp = ExperimentsGKAN(config, G, n_trials, method, t_f_train=240)
    exp.run()
    
    
if __name__ == '__main__':
    run('./configs/config_kuramoto.yml', method='optuna', n_trials=45)
    
    
    