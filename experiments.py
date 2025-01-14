from utils.utils import load_config
from utils.model_selection import ModelSelector
import networkx as nx


def run(config_path, n_trials=10, method='grid_search'):
    config = load_config(config_path)
    noise_strengths = config["noise_strengths"]
    
    if len(noise_strengths)==0:
        _run(config, n_trials=n_trials, method=method)
    else:
        for noise_strength in noise_strengths:
            _run(config, noise_strength, n_trials=n_trials, method=method)
    
    
def _run(config, noise_level=None, n_trials=10, method='grid_search'):
    
    G = nx.grid_2d_graph(7, 10)
    model_selector = ModelSelector(config=config, G=G, noise_level=noise_level, n_trials=n_trials, method=method)
    best_params = model_selector.optimize()
    model_selector.eval_model(best_params=best_params)


if __name__ == '__main__':
    run('./configs/config_neuronal.yml')