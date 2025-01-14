import torch
from torch_geometric.data import Data
import yaml

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os 
import numpy as np

from datasets.dynamical_datasets import GraphDynamics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def save_logs(file_name, log_message, save_updates=True):
    if save_updates:
        print(log_message)
        with open(file_name, 'a') as logs:
            logs.write('\n'+log_message)
            

            
def sample_temporal_graph(dataset, device='cpu', sample_size=32):
    '''
    Returns a sample from the given time-series consisting of graph snapshots at different time steps.
    '''
    interval = len(dataset) // sample_size
    sampled_indices = torch.tensor([i * interval for i in range(sample_size)])

    data_samples = dataset[sampled_indices]
    num_nodes = data_samples[0].x.shape[0]  # Number of nodes in each graph

    samples = [d.x for d in data_samples]
    ys = [d.y for d in data_samples]
    
    concatenated_x = torch.cat(samples, dim=0)  # Shape: (sample_size * num_nodes, in_dim)
    concatenated_ys = torch.cat(ys, dim=0)

    all_edges = []
    for i, d in enumerate(data_samples):
        offset = i * num_nodes
        edge_index = d.edge_index + offset
        all_edges.append(edge_index)

    concatenated_edge_index = torch.cat(all_edges, dim=1)

    dummy_x = Data(
        edge_index=concatenated_edge_index,
        x=concatenated_x,
        y = concatenated_ys,
        delta_t=1
    )

    dummy_x = dummy_x.to(device)
    
    return dummy_x


def load_config(config_path='config.yml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_temporal_data_loader(dataset, device, batch_size):
    collate_fn = lambda batch, dev: [snapshot.to(dev) for snapshot in batch]
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, device), shuffle=False)
    return loader


@torch.no_grad()
def get_acts(model, dummy_x = None):
    store_act = model.store_act
    model.store_act = True
    _ = model.forward(dummy_x)
    model.store_act = store_act
    

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
                
                

def get_conf(config, noise_strength, rng):
    conf = {
        'dynamics': config['dynamics'],
        'time_steps': config.get('time_steps', 1000),
        'num_samples': config.get('num_samples', 1000),
        'step_size': config.get('step_size', 0.01),
        'initializer': config.get('initializer', lambda N: np.random.random(N)),
        'min_sample_distance': config.get('min_sample_distance', 1),
        'rng': rng,
        'regular_samples': config.get('regular_samples', True),
        'add_noise_target': config.get('add_noise_target', False),
        'noise_strength': noise_strength,
        'add_noise_input': config.get('add_noise_input', False),
        'in_noise_dim': config.get('in_noise_dim', 1),
        **config.get('integration_kwargs', {})
    }
    
    return conf


def create_datasets(conf, root, name, graph):
    conf['graph'] = graph
    conf['root'] = root
    conf['name'] = name
    
    conf['name_suffix'] = 'train'
    train_dataset = GraphDynamics(**conf)
    
    conf['name_suffix'] = 'valid'
    conf['time_steps'] //= 2
    conf['num_samples'] //= 2
    valid_dataset = GraphDynamics(**conf)
    
    conf['name_suffix'] = 'test'
    test_dataset = GraphDynamics(**conf)
    
    return train_dataset, valid_dataset, test_dataset


def save_acts(layers, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for l, layer in enumerate(layers):
        assert layer.cache_act is not None and layer.cache_preact is not None, 'Populate model activations before saving them'
        torch.save(layer.cache_preact, f"{folder_path}/cache_preact_{l}")
        torch.save(layer.cache_act, f"{folder_path}/cache_act_{l}")
    
    
    
    