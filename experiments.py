from utils.utils import load_config, save_acts, plot, sample_from_spatio_temporal_graph, create_datasets, pre_processing
from utils.model_selection import ModelSelector
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from models.NetWrapper import NetWrapper
from models.KanGDyn import KanGDyn
from train_and_eval import fit


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
    # run('./configs/config_kuramoto.yml')
    config = load_config(config_path='./configs/config_kuramoto.yml')
    G = nx.complete_graph(10)
    # G = nx.grid_2d_graph(7, 10)
    train_data, t_train, valid_data, t_valid , test_data, t_test = create_datasets(config=config, graph=G)
    
    train_data = pre_processing(train_data)
    valid_data = pre_processing(valid_data)
    test_data = pre_processing(test_data)
    
    edge_index = from_networkx(G).edge_index
    
    model_config = {
        'h_hidden_layers':[2, 1],
        'g_hidden_layers':[2,1,1],
        'grid_range':[-3, 3],
        'grid_size': 7,
        'model_path':'./saved_models/higher-order-kuramoto',
        'device':'cuda',
        'store_acts':True
    }
    
    model = NetWrapper(KanGDyn, model_config, edge_index, update_grid=False)
    
    criterion = torch.nn.MSELoss()
    lr = 0.001
    mu_1 = 1.
    mu_2 = 1.
    lmbd = 0.0001
    epochs = 300
    
    _ = fit(
        model,
        train_data,
        t_train,
        valid_data,
        t_valid,
        test_data,
        t_test,
        epochs=epochs,
        patience=100,
        lr=lr,
        lmbd=lmbd,
        mu_1=mu_1,
        mu_2=mu_2,
        criterion=criterion,
        use_orig_reg=True,
        save_updates=True,
        opt='Adam'
    )
    
    best_model_state = torch.load(f'./{model.model.model_path}/best_state_dict.pth', weights_only=False)
    
    model.load_state_dict(best_model_state)

    net = model.model

    net.h_net.store_act = True
    net.g_net.store_act = True

    dummy_x, dummy_edge_index = sample_from_spatio_temporal_graph(train_data, edge_index, sample_size=32)

    with torch.no_grad():
        _ = net(dummy_x, dummy_edge_index)

    plot(folder_path=f'{net.h_net.model_path}/figures', layers=net.h_net.layers, show_plots=False)
    plot(folder_path=f'{net.g_net.model_path}/figures', layers=net.g_net.layers, show_plots=False)

    save_acts(layers=net.h_net.layers, folder_path=f'{net.h_net.model_path}/cached_acts')
    save_acts(layers=net.g_net.layers, folder_path=f'{net.g_net.model_path}/cached_acts')
    
    
    