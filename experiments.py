from utils.utils import load_config, save_acts, plot, sample_from_spatio_temporal_graph
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
    edge_index = model_selector.edge_index
    
    net = model.model

    net.h_net.store_act = True
    net.g_net.store_act = True

    dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(model_selector.train_data, 
                                                                  edge_index, 
                                                                  sample_size=32)

    with torch.no_grad():
        _ = net(dummy_x, dummy_edge_index)

    plot(folder_path=f'{net.h_net.model_path}/figures', layers=net.h_net.layers, show_plots=False)
    plot(folder_path=f'{net.g_net.model_path}/figures', layers=net.g_net.layers, show_plots=False)

    save_acts(layers=net.h_net.layers, folder_path=f'{net.h_net.model_path}/cached_acts')
    save_acts(layers=net.g_net.layers, folder_path=f'{net.g_net.model_path}/cached_acts')
    


if __name__ == '__main__':
    run('./configs/config_kuramoto.yml')