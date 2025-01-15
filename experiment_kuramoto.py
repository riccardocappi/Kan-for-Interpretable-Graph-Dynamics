from utils.utils import load_config, get_conf, create_datasets, sample_temporal_graph, get_acts, plot, save_acts
import torch
import numpy as np
import networkx as nx
from train_and_eval import fit
from models.KanGDyn import KanGDyn


def main():
    config_path = './configs/config_kuramoto.yml'
    config = load_config(config_path)
    
    root = config["root"]
    name = config["name"]
    epsilon = config['step_size']
    
    device = config["device"]
    if device == 'cuda':
        assert torch.cuda.is_available()
        
    rng = np.random.default_rng(seed=config.get("seed", 42))
    conf = get_conf(config, None, rng)
    
    # G = nx.grid_2d_graph(7, 10)
    # G = nx.complete_graph(10)
    
    G = nx.erdos_renyi_graph(60, 0.1)
    train_dataset, valid_dataset, test_dataset = create_datasets(conf, root=root, name=name, graph=G)
    
    model = KanGDyn(
        h_hidden_layers=[2,1],
        g_hidden_layers=[2,1,1],
        grid_range=[0,1],
        model_path='./saved_models/model-kuramoto',
        epsilon=epsilon,
        device=device,
        store_acts=False,
        norm=False
    )
    
    _ = fit(model,
            train_dataset,
            valid_dataset,
            test_dataset,
            epochs=150,
            criterion=torch.nn.L1Loss(),
            lamb=0.,
            lr=0.01,
            use_orig_reg=False
            )
    
    
    dummy_x = sample_temporal_graph(train_dataset, device=device, sample_size=32)
    model.h_net.store_act = True
    model.g_net.store_act = True
    get_acts(model, dummy_x)
    
    plot(folder_path=f'{model.h_net.model_path}/figures', layers=model.h_net.layers, show_plots=False)
    plot(folder_path=f'{model.g_net.model_path}/figures', layers=model.g_net.layers, show_plots=False)
    
    save_acts(layers=model.h_net.layers, folder_path=f'{model.h_net.model_path}/cached_acts')
    save_acts(layers=model.g_net.layers, folder_path=f'{model.g_net.model_path}/cached_acts')
    
    torch.save(dummy_x, f"{model.model_path}/sample")


if __name__ == '__main__':
    main()
    
    
    