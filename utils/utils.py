from pysr import PySRRegressor
import torch
import yaml
import matplotlib.pyplot as plt
import os 
import numpy as np
from datasets.data_utils import numerical_integration
import sympy as sp
import dill



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datasets.GraphDynamics import GraphDynamics



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
                plt.close()
                


def integrate(config, graph, rng):
    N = graph.number_of_nodes()
    input_range = config['input_range']
    t_span = config['t_span']
    y0 = rng.uniform(input_range[0], input_range[1], N)
    t_eval_steps = config['t_eval_steps']
    dynamics = config['dynamics']
    device = torch.device(config['device'])
    
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
    concatenated_x = torch.reshape(samples, (-1, samples.size(2)))
    concatenated_x = concatenated_x.to(device)
    
    all_edges = []
    num_nodes = dataset.size(1)
    for i in range(len(samples)):
        offset = i * num_nodes
        upd_edge_index = edge_index + offset
        all_edges.append(upd_edge_index) 
        
    concatenated_edge_index = torch.cat(all_edges, dim=1)
    concatenated_edge_index = concatenated_edge_index.to(device)
    
    return concatenated_x, concatenated_edge_index
    
  
    
def create_datasets(config, graph, t_f_train=100):
    rng = np.random.default_rng(seed=config['seed'])
    
    data, t = [], []
    for _ in range(config['n_iter']):
        data_k, t_k = integrate(config, graph, rng)
        data.append(data_k)
        t.append(t_k)
        
    data = torch.stack(data, dim=0)
    t = torch.stack(t, dim=0)

    train_data = data[:, :t_f_train, :, :]
    t_train = t[:, :t_f_train]
    
    training_set = GraphDynamics(train_data, t_train)
    validation_set = GraphDynamics(data, t)
    
    # return train_data, t_train, data, t
    return training_set, validation_set
    

def save_acts(layers, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for l, layer in enumerate(layers):
        assert layer.cache_act is not None and layer.cache_preact is not None, 'Populate model activations before saving them'
        torch.save(layer.cache_preact, f"{folder_path}/cache_preact_{l}")
        torch.save(layer.cache_act, f"{folder_path}/cache_act_{l}")
        torch.save(layer.acts_scale_spline, f"{folder_path}/cache_act_scale_spline_{l}")
        
        

def save_black_box_to_file(folder_path, cache_input, cache_output):    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    torch.save(cache_input, f'{folder_path}/cached_input')
    torch.save(cache_output, f'{folder_path}/cached_output')
 
        

def pruning(kan_acts, kan_preacts, theta = 0.01):

    def get_acts_scale_spline(l_index):
        input_range = torch.std(pruned_preacts[l_index], dim=0)
        output_range_spline = torch.std(pruned_acts[l_index], dim=0)
        acts_scale_spline = output_range_spline / input_range
        return acts_scale_spline

    n_layers = len(kan_acts)
    pruned_acts = kan_acts.copy()
    pruned_preacts = kan_preacts.copy()

    for l in range(n_layers-1):
        acts_scale_spline = get_acts_scale_spline(l)
        I_lj, _ = torch.max(acts_scale_spline, dim=1)

        acts_scale_spline_next = get_acts_scale_spline(l+1)
        O_lj, _ = torch.max(acts_scale_spline_next, dim=0)

        pruned_nodes = ((I_lj < theta) | (O_lj < theta)).bool()
        remaining_indices = torch.where(~pruned_nodes)[0]
        remaining_acts = pruned_acts[l][:, remaining_indices, :]

        pruned_acts[l] = remaining_acts
        pruned_acts[l+1] = pruned_acts[l+1][:, :, remaining_indices]
        pruned_preacts[l+1] = pruned_preacts[l+1][:, remaining_indices]

        for j, is_pruned in enumerate(pruned_nodes):
            if is_pruned:
                print(f"Pruning node ({l},{j})")

    return pruned_acts, pruned_preacts



def fit_acts_pysr(x, y):
    model = PySRRegressor(
        niterations=100,  # Number of iterations
        unary_operators=[
        "exp",
        "sin",
        "neg",
        "square",
        "cube",
        "abs",
        "log",
        "log10",
        "sqrt",
        "tan",
        "tanh",
        "zero(x) = 0*x"
        ],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-4 && complexity < 5"
            # Stop early if we find a good and simple equation
            ),
        maxsize=7,
        maxdepth=5,
        verbosity=0,
        extra_sympy_mappings={"zero": lambda x: 0*x},
        delete_tempfiles=True,
        temp_equation_file=True,
        tempdir='./pysr'

    )
    model.fit(x, y)
    best_equation = model.equations_.nlargest(1, 'score').iloc[0]['sympy_format']
    sympy_function = sp.sympify(best_equation)

    return sympy_function
    # return model.equations_.nlargest(5, 'score')


def fit_layer(cached_act, cached_preact, symb_xs, device='cpu'):
    symb_layer_acts = []
    in_dim = cached_act.size(2)
    out_dim = cached_act.size(1)
    symbolic_functions = [[lambda x: 0*x for _ in range(in_dim)] for _ in range(out_dim)]

    for j in range(out_dim):
        symb_out = 0
        for i in range(in_dim):
            x = cached_preact[:, i].cpu().detach().numpy().reshape(-1, 1)
            y = cached_act[:, j, i].cpu().detach().numpy().reshape(-1, 1)
            symb_func = fit_acts_pysr(x, y)
            # pdb.set_trace()

            symbolic_functions[j][i] = sp.lambdify(sp.Symbol('x0'), symb_func)

            symb_func = symb_func.subs(sp.Symbol('x0'), symb_xs[i])
            symb_out += symb_func

        symb_layer_acts.append(symb_out)

    return symb_layer_acts, symbolic_functions


def fit_kan(kan_acts, kan_preacts, symb_xs, save_path='./symb_functions'):
    n_layers = len(kan_acts)
    all_functions = []
    for l in range(n_layers):
        acts = kan_acts[l]
        preacts = kan_preacts[l]

        symb_xs, symb_functions = fit_layer(
            cached_act=acts,
            cached_preact=preacts,
            symb_xs=symb_xs,
            device='cpu'
        )
        all_functions.append(symb_functions)

    with open(save_path, "wb") as f:
        dill.dump(all_functions, f)

    return symb_xs
                
        
def load_cached_data(cached_acts_path, cached_preacts_path, device='cpu'):
    cached_act = torch.load(cached_acts_path, weights_only=False, map_location=torch.device(device)) # (batch_dim, in_dim)
    cached_preact = torch.load(cached_preacts_path, weights_only=False, map_location=torch.device(device)) # (batch_dim, out_dim, in_dim)
    return cached_act, cached_preact


def get_kan_arch(n_layers, model_path):
    act_name_prefix = 'cache_act'
    preact_name_prefix = 'cache_preact'
    acts, preacts = [], []
    for l in range(n_layers):
        cached_acts, cached_preacts = load_cached_data(
            cached_acts_path = f'{model_path}/cached_acts/{act_name_prefix}_{l}',
            cached_preacts_path = f'{model_path}/cached_acts/{preact_name_prefix}_{l}'
        )
        acts.append(cached_acts)
        preacts.append(cached_preacts)

    return acts, preacts


def fit_model(n_h_hidden_layers, n_g_hidden_layers, model_path, theta=0.1):
    # G_net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/G_Net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)
    symb_g = fit_kan(pruned_acts,
                     pruned_preacts,
                     symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')],
                     save_path=f'{model_path}/G_Net/symb_functions'
                     )

    # H_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_h_hidden_layers, model_path=f'{model_path}/H_Net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)
    symb_h_in = [sp.Symbol('x_i'), sp.Symbol(r'\sum_j( ' + str(symb_g[0]) + ')')]
    symb_h = fit_kan(pruned_acts,
                     pruned_preacts,
                     symb_xs=symb_h_in,
                     save_path=f'{model_path}/H_Net/symb_functions'
                     )

    return symb_h


def fit_black_box(cached_input, cached_output, symb_xs):
    in_dim = cached_input.size(1)
    out_dim = cached_output.size(1)

    x = cached_input.detach().numpy().reshape(-1, in_dim)
    y = cached_output.detach().numpy().reshape(-1, out_dim)

    symb_func = fit_acts_pysr(x, y)
    # pdb.set_trace()

    subs_dict = {sp.Symbol(f'x{i}'): symb_xs[i] for i in range(len(symb_xs))}

    symb_func = symb_func.subs(subs_dict)

    return symb_func



def fit_mpnn(model_path, device='cpu'):
    # G_Net
    cached_input = torch.load(f'{model_path}/G_net/cached_data/cached_input', weights_only=False, map_location=torch.device(device))
    cached_output = torch.load(f'{model_path}/G_net/cached_data/cached_output', weights_only=False, map_location=torch.device(device))
    symb_g = fit_black_box(cached_input, cached_output, symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')])

    # H_Net
    cached_input = torch.load(f'{model_path}/H_net/cached_data/cached_input', weights_only=False, map_location=torch.device(device))
    cached_output = torch.load(f'{model_path}/H_net/cached_data/cached_output', weights_only=False, map_location=torch.device(device))

    symb_h_in = [sp.Symbol('x_i'), sp.Symbol(r'\sum_{j}( ' + str(symb_g) + ')')]
    symb_h = fit_black_box(cached_input, cached_output, symb_xs=symb_h_in)

    return symb_h


def fit_black_box_from_kan(model_path, n_g_hidden_layers, n_h_hidden_layers, device='cpu', theta=0.1):
    #G_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/G_Net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)

    input = pruned_preacts[0]
    output = pruned_acts[-1].sum(dim=2)

    symb_g = fit_black_box(input, output, symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')])

    #H_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_h_hidden_layers, model_path=f'{model_path}/H_Net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)

    input = pruned_preacts[0]
    output = pruned_acts[-1].sum(dim=2)

    symb_h_in = [sp.Symbol('x_i'), sp.Symbol(r'\sum_{j}( ' + str(symb_g) + ')')]

    symb_h = fit_black_box(input, output, symb_xs=symb_h_in)

    return symb_h