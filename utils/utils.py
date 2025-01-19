import torch
import yaml
import matplotlib.pyplot as plt
import os 
import numpy as np
from datasets.data_utils import numerical_integration

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def save_logs(file_name, log_message, save_updates=True):
    if save_updates:
        print(log_message)
        with open(file_name, 'a') as logs:
            logs.write('\n'+log_message)



def load_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def plot(folder_path, layers, show_plots=False):
    '''
    Plots the shape of all the activation functions
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for l, layer in enumerate(layers):
        assert layer.cache_act is not None and layer.cache_preact is not None, 'Populate model activations before plotting' 
        activations = layer.cache_act
        pre_activations = layer.cache_preact
        preact_sorted, indices = torch.sort(pre_activations, dim=0)
        for j in range(layer.out_features):
            for i in range(layer.in_features):
                out = activations[:, j, i]
                out = out[indices[:, i]]
                plt.figure()
                plt.plot(preact_sorted[:, i].cpu().detach().numpy(), out.cpu().detach().numpy())
                plt.title(f"Act. (Layer: {l}, Neuron: {j}, Input: {i})")
                plt.savefig(f"{folder_path}/out_{l}_{j}_{i}.png")
                if show_plots:
                    plt.show()
                plt.clf()
                


def integrate(config, graph):
    seed = config['seed']
    rng = np.random.default_rng(seed=seed)
    N = graph.number_of_nodes()
    input_range = config['input_range']
    t_span = config['t_span']
    y0 = rng.uniform(input_range[0], input_range[1], N)
    t_eval_steps = config['t_eval_steps']
    dynamics = config['dynamics']
    device = config['device']
    
    xs, t = numerical_integration(
        G=graph,
        dynamics=dynamics,
        initial_state=y0,
        time_span=t_span,
        t_eval_steps=t_eval_steps,
        **config.get('integration_kwargs', {})
    )
    xs = np.transpose(xs)
    return torch.from_numpy(xs).float().unsqueeze(2).to(device), torch.from_numpy(t).float().to(device)



def sample_from_spatio_temporal_graph(dataset, edge_index, sample_size=32):
    device = dataset.device
    interval = len(dataset) // sample_size
    sampled_indices = torch.tensor([i * interval for i in range(sample_size)])
    samples = dataset[sampled_indices]
    concatenated_x = torch.reshape(samples, (-1, samples.size(2))).to(device)
    
    all_edges = []
    num_nodes = dataset.size(1)
    for i,s in enumerate(samples):
        offset = i * num_nodes
        upd_edge_index = edge_index + offset
        all_edges.append(upd_edge_index) 
        
    concatenated_edge_index = torch.cat(all_edges, dim=1).to(device)
    
    return concatenated_x, concatenated_edge_index
    
  
    
def create_datasets(config, graph):
    train_data, t_train = integrate(config, graph)
    
    config['t_eval_steps'] //= 2

    valid_data, t_valid = integrate(config, graph)
    test_data, t_test = integrate(config, graph)
    
    return train_data, t_train, valid_data, t_valid, test_data, t_test
    
    

def save_acts(layers, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for l, layer in enumerate(layers):
        assert layer.cache_act is not None and layer.cache_preact is not None, 'Populate model activations before saving them'
        torch.save(layer.cache_preact, f"{folder_path}/cache_preact_{l}")
        torch.save(layer.cache_act, f"{folder_path}/cache_act_{l}")  