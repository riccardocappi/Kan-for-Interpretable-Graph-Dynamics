from utils.utils import load_config, get_acts, save_acts, plot, sample_temporal_graph
from utils.model_selection import ModelSelector
import networkx as nx
import torch



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
    # G = nx.complete_graph(10)
    model_selector = ModelSelector(config=config, G=G, noise_level=noise_level, n_trials=n_trials, method=method)
    best_params = model_selector.optimize()
    
    model = model_selector.eval_model(best_params=best_params)
    
    dummy_x = sample_temporal_graph(model_selector.train_dataset, device=model.device, sample_size=32)
    model.h_net.store_act = True
    model.g_net.store_act = True
    get_acts(model, dummy_x)
    
    plot(folder_path=f'{model.h_net.model_path}/figures', layers=model.h_net.layers, show_plots=False)
    plot(folder_path=f'{model.g_net.model_path}/figures', layers=model.g_net.layers, show_plots=False)
    
    save_acts(layers=model.h_net.layers, folder_path=f'{model.h_net.model_path}/cached_acts')
    save_acts(layers=model.g_net.layers, folder_path=f'{model.g_net.model_path}/cached_acts')
    
    torch.save(dummy_x, f"{model.model_path}/sample")
    


if __name__ == '__main__':
    run('./configs/config_kuramoto.yml')