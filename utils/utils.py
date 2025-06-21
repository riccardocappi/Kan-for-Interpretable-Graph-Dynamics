"""
Some static methods
"""


from pysr import PySRRegressor
import torch
import yaml
import matplotlib.pyplot as plt
import os 
import numpy as np
from datasets.data_utils import numerical_integration
import sympy as sp
import json
from models.kan.KanLayer import KANLayer
from models.kan.KAN import KAN
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sympy import count_ops
import warnings
import sympy
import re
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore", category=FutureWarning)
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)


SCORES = {
    'MSE': torch.nn.MSELoss(),
    'MAE': torch.nn.L1Loss()
}


SYMBOLIC_LIB_NUMPY = {
    'x': lambda x: x,
    'x^2': lambda x: x**2,
    'x^3': lambda x: x**3,
    'exp': lambda x: np.clip(np.exp(x), a_min=None, a_max=1e5),
    'abs': lambda x: np.abs(x),
    'sin': lambda x: np.sin(x),
    'cos': lambda x: np.cos(x),
    'tan': lambda x: np.tan(x),
    'tanh': lambda x: np.tanh(x),
    'ln': lambda x: np.log(x),
    '0': lambda x: x*0,
}

SYMBOLIC_LIB_SYMPY = {
    'x': lambda x: x,
    'x^2': lambda x: x**2,
    'x^3': lambda x: x**3,
    'exp': lambda x: sp.exp(x),
    'abs': lambda x: sp.Abs(x),
    'sin': lambda x: sp.sin(x),
    'cos': lambda x: sp.cos(x),
    'tan': lambda x: sp.tan(x),
    'tanh': lambda x: sp.tanh(x),
    'ln': lambda x: sp.ln(x),
    '0': lambda x: 0 * x,
}


def save_logs(file_name, log_message, save_updates=True):
    """
    Save logs to file
    
    Args:
        - file_name : Logs file name
        - log_message : Message to save
        - save_updates : Whether to save log message or not
    """
    if save_updates:
        print(log_message)
        with open(file_name, 'a') as logs:
            logs.write('\n'+log_message)



def load_config(config_path='config.yml'):
    """
    Returns a dictionary of the specified config file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def plot(folder_path, layers, show_plots=False):
    '''
    Plots the shape of all the activation functions of the specified KAN layer
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
                plt.plot(preact_sorted[:, i].cpu().detach().numpy(), out.cpu().detach().numpy(), linewidth=2.5)
                plt.title(f"Act. (Layer: {l}, Neuron: {j}, Input: {i})")
                plt.savefig(f"{folder_path}/out_{l}_{j}_{i}.png")
                if show_plots:
                    plt.show()
                plt.clf()
                plt.close()
                

def fix_symbolic(layer:KANLayer, i, j, func):
    """
    Fix KAN spline to symbolic function
    """
    layer.symbolic_functions[j][i] = func
    layer.layer_mask.data[j][i] = 0
    layer.symb_mask.data[j][i] = 1
                
                
def automatic_fix_symbolic_kan(symb_functions_file, in_dim=1, device='cuda'):
    """
    Automatically fix all the KAN splines to the respective symbolic functions
    """
    with open(symb_functions_file, "r") as f:
        all_functions = json.load(f)
    
    hidden_layers = [in_dim]
    # Loop over layers
    layer_keys = sorted(all_functions.keys(), key=int)
    for layer_key in layer_keys:
        layer = all_functions[layer_key]
        num_nodes = len(layer)
        hidden_layers.append(num_nodes)
    
    if hidden_layers[-1] == 0:
        return lambda x: 0 * x[:, 0].unsqueeze(-1)
    
    kan_placeholder = KAN(
        layers_hidden=hidden_layers,
        store_act=True,
        compute_symbolic=True,
        device=device
    )
    
    for l, layer in enumerate(kan_placeholder.layers):
        symb_layer = all_functions[f"{l}"]
        for j in range(layer.out_features):
            for i in range(layer.in_features):
                str_func =  symb_layer[f"{j}"][f"{i}"]
                symb_func = sp.lambdify(sp.Symbol('x0'), sp.sympify(str_func))
                fix_symbolic(layer, i, j, symb_func)
    return kan_placeholder
                


def integrate(
    input_range,
    t_span,
    t_eval_steps,
    dynamics,
    device,   
    graph,
    rng,
    **integration_kwargs
):
    """
    Integrates the specified dynamics over the given graph
    """
    N = graph.number_of_nodes()
    y0 = rng.uniform(input_range[0], input_range[1], N).astype(np.float64)

    xs, t = numerical_integration(
        G=graph,
        dynamics=dynamics,
        initial_state=y0,
        time_span=t_span,
        t_eval_steps=t_eval_steps,
        **integration_kwargs
    )
    return torch.from_numpy(xs).float().unsqueeze(2).to(device), torch.from_numpy(t).float().to(device)


def sample_from_spatio_temporal_graph(dataset, edge_index, edge_attr, t=None, sample_size=32):
    device = dataset.device
    
    sample_size = sample_size if sample_size != -1 else len(dataset)
    interval = len(dataset) // sample_size
    sampled_indices = torch.tensor([i * interval for i in range(sample_size)], device=device)
    
    samples = dataset[sampled_indices]
    t_sampled = t[sampled_indices] if t is not None else torch.tensor([], device=device)
    concatenated_x = torch.reshape(samples, (-1, samples.size(2))).to(device)
    
    concatenated_t = t_sampled.unsqueeze(0).repeat(dataset.size(1), 1).reshape(-1, 1)
    
    all_edges = []
    all_edge_attrs = []
    num_nodes = dataset.size(1)
    
    for i in range(sample_size):
        offset = i * num_nodes
        upd_edge_index = edge_index + offset
        all_edges.append(upd_edge_index)
        
        if edge_attr is not None:
            all_edge_attrs.append(edge_attr.clone())  # Clone in case attributes are mutable
    
    concatenated_edge_index = torch.cat(all_edges, dim=1).to(device)
    
    if edge_attr is not None:
        concatenated_edge_attr = torch.cat(all_edge_attrs, dim=0).to(device)
    else:
        concatenated_edge_attr = None

    return concatenated_x, concatenated_edge_index, concatenated_t, concatenated_edge_attr
    

def sample_irregularly_per_ics(data, time, num_samples):
    ics, n_step, n_nodes, in_dim = data.shape
    num_samples = num_samples if num_samples > 0 else n_step
    sampled_data = torch.zeros((ics, num_samples, n_nodes, in_dim), dtype=data.dtype, device=data.device)
    sampled_times = torch.zeros((ics, num_samples), dtype=time.dtype, device=time.device)
    sampled_indices = torch.zeros((ics, num_samples), dtype=torch.int32, device=data.device)

    for i in range(ics):
        indices = torch.randperm(n_step)[:num_samples]  # Random unique indices
        indices = torch.sort(indices).values  # Optional: Sort indices to maintain order
        
        sampled_data[i] = data[i, indices, :, :]  # Sample separately for each ics
        sampled_times[i] = time[i, indices]
        sampled_indices[i] = indices

    return sampled_data, sampled_times


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
        input_range = torch.sum(torch.abs(pruned_preacts[l_index]), dim=0)
        output_range_spline = torch.sum(torch.abs(pruned_acts[l_index]), dim=0)
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



def get_pysr_model(
    n_iterations=100,
    binary_operators = ['+', '-', '*', '/'], 
    extra_sympy_mappings = {},
    unary_operators = None,
    **kwargs):
    
    extra_mapping = {"zero": lambda x: x*0}
    extra_mapping.update(extra_sympy_mappings)
    
    if unary_operators is None:
        unary_operators = [
            "exp",
            "sin",
            "neg",
            "square",
            "cube",
            "abs",
            "tan",
            "tanh",
            "log",
            "log1p",
            "zero(x) = 0*x"
        ]
    
    model = PySRRegressor(
        niterations=n_iterations,  # Number of iterations
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        maxsize=7,
        maxdepth=5,
        verbosity=0,
        extra_sympy_mappings=extra_mapping,
        delete_tempfiles=True,
        temp_equation_file=True,
        tempdir='./pysr',
        progress=False,
        **kwargs
    )
    
    return model
    


def fit_acts_pysr(x, y, pysr_model = None, sample_size = -1, seed=42):
    rng = np.random.default_rng(seed)
    if pysr_model is None:
        model = get_pysr_model()
    else:
        model = pysr_model()
    
    if sample_size > 0 and sample_size < len(x):
        indices = rng.choice(len(x), sample_size, replace=False)
        x_sampled = x[indices]
        y_sampled = y[indices]
    else:
        x_sampled = x
        y_sampled = y
    
    model.fit(x_sampled, y_sampled)
    top_5_eq = model.equations_.nlargest(5, 'score')
    return top_5_eq 


def penalized_loss(y_true, y_pred, func_symb, alpha=0.01):
    mse = mean_squared_error(y_true, y_pred)
    complexity = count_ops(func_symb) 
    penalty = alpha * complexity
    return mse + penalty


def fit_params_scipy(x_train, y_train, func, func_name, alpha=0.1):  
    if func_name == 'x' or func_name == 'neg':
        func_optim = lambda x, a, b: a*x + b  
        init_params = [1., 0.]
    elif func_name=='x^2':
        func_optim = lambda x, a, b, c: a*x**2 + b*x + c  
        init_params = [1., 0., 0]
    elif func_name == 'x^3':
        func_optim = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d  
        init_params = [1., 0., 0., 0.]
    elif (func_name == 'ln') and (np.any(x_train <= 0)):
        return 1e8, [], 0, lambda x: x*0
    else:
        func_optim = lambda x, a, b, c, d: c * func(a*x + b) + d
        init_params = [1., 0., 1., 0.]
        
    try:
        params, _ = curve_fit(func_optim, x_train, y_train, p0=init_params, nan_policy='omit')
    except RuntimeError:
        return 1e8, [], 0, lambda x: x*0
    
    x_symb = sp.Symbol('x0')    # This symbol must be x0 in order to work with the rest of the code
    
    if func_name == 'x' or func_name == 'neg':
        post_fun = params[0] * func(x_train) + params[1]
        fun_sympy = params[0] * x_symb + params[1]
    elif func_name == 'x^2':
        post_fun = params[0] * x_train**2 + params[1] * x_train + params[2]
        fun_sympy = params[0] * x_symb**2 + params[1] * x_symb + params[2]
    elif func_name == 'x^3':
        post_fun = params[0]*x_train**3 + params[1]*x_train**2 + params[2]*x_train + params[3] 
        fun_sympy = params[0] * x_symb**3 + params[1] * x_symb**2 + params[2] * x_symb + params[3]
    else:
        post_fun = params[2] * func(params[0]*x_train + params[1]) + params[3]
        fun_sympy = params[2] * SYMBOLIC_LIB_SYMPY[func_name](params[0] * x_symb + params[1]) + params[3]
    
    
    if np.any(np.isnan(post_fun)) or np.any(np.isinf(post_fun)):
        return 1e8, [], 0, lambda x: x*0
    
    fun_sympy_quantized = quantise(fun_sympy, 1e-3)
    mse = penalized_loss(y_train, post_fun, fun_sympy_quantized, alpha=alpha)
    return mse, params, fun_sympy_quantized, func_optim


def fit_acts_scipy(x, y, alpha=0.1):    
    scores = []
    for name, func in SYMBOLIC_LIB_NUMPY.items():
        mse, params, symb, func_optim = fit_params_scipy(x, y, func, name, alpha=alpha)
        scores.append((symb, mse, params, func_optim))
    
    best_fun_sympy, _, best_params, best_func_optim  = min(scores, key=lambda x: x[1])    
    return best_fun_sympy, best_params, best_func_optim


def find_best_symbolic_func(x_train, y_train, x_val, y_val, alpha_grid):
    results = []

    for alpha in alpha_grid:
        symb_func, params, func_optim = fit_acts_scipy(x_train, y_train, alpha=alpha)
        val_mse = mean_squared_error(y_val, func_optim(x_val, *params))
        complexity = count_ops(symb_func)
        log_loss = np.log(val_mse)
        results.append((symb_func, complexity, log_loss, alpha))

    # Sort by complexity to compute finite difference derivative
    results.sort(key=lambda x: x[1])  # sort by complexity
    top_equations = pd.DataFrame(results, columns=["symbolic_function", "complexity", "log_loss", "alpha"])
    scores = [(results[0][0], 0)]

    for k in range(1, len(results)):
        c2, c1 = results[k][1], results[k - 1][1]
        l2, l1 = results[k][2], results[k - 1][2]
        
        if c1==c2: continue

        dlogloss_dcomplexity = (l2 - l1) / (c2 - c1)
        score = -dlogloss_dcomplexity
        scores.append((results[k][0], score))

    best_symb_func, _ = max(scores, key=lambda x: x[1])
    return best_symb_func, str(best_symb_func), top_equations


def fit_layer(cached_act, cached_preact, symb_xs, val_ratio=0.2, seed=42, model_path='./models'):
    alpha_grid = torch.logspace(-5, -1, steps=5)
    
    symb_layer_acts = []
    symbolic_functions = defaultdict(dict)
    top_equations = {}

    in_dim = cached_act.shape[2]
    out_dim = cached_act.shape[1]

    for j in range(out_dim):
        symb_out = 0

        for i in range(in_dim):
            x = cached_preact[:, i].reshape(-1)
            y = cached_act[:, j, i].reshape(-1)

            x_train, x_val, y_train, y_val =  train_test_split(x, y, test_size=val_ratio, random_state=seed)

            best_symb_func, best_func_str, top_eq = find_best_symbolic_func(
                x_train, y_train, x_val, y_val,
                alpha_grid=alpha_grid
            )
            top_equations[(i, j)] = top_eq

            symbolic_functions[j][i] = best_func_str
            best_symb_func = best_symb_func.subs(sp.Symbol('x0'), symb_xs[i])
            symb_out += best_symb_func
            
            

        symb_layer_acts.append(symb_out)

    return symb_layer_acts, symbolic_functions, top_equations


def fit_kan(kan_acts, kan_preacts, symb_xs, model_path='./models'):
    n_layers = len(kan_acts)
    all_functions = {}
    top_5_save_path = f"{model_path}/top_eqs"
    os.makedirs(top_5_save_path, exist_ok=True)
    
    for l in range(n_layers):
        acts = kan_acts[l].cpu().detach().numpy()
        preacts = kan_preacts[l].cpu().detach().numpy()

        symb_xs, symb_functions, top_equations = fit_layer(
            cached_act=acts,
            cached_preact=preacts,
            symb_xs=symb_xs
        )
        
        all_functions[l] = symb_functions
        
        for k, df in top_equations.items():
            df.to_csv(f"{top_5_save_path}/top_equations({l}, {k[1]}, {k[0]}).csv")
        
    save_path = f"{model_path}/symb_functions.json"
    with open(save_path, "w") as f:
        json.dump(all_functions, f)
            
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


def fit_model(n_h_hidden_layers, n_g_hidden_layers, model_path, theta=0.1, message_passing=True, include_time=False):
    # G_net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/g_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)    
    
    symb_g = fit_kan(
        pruned_acts,
        pruned_preacts,
        symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')],
        model_path=f"{model_path}/g_net"
    )
    symb_g = symb_g[0]  # Univariate functions
    # H_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_h_hidden_layers, model_path=f'{model_path}/h_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)
    
    aggr_term = sp.Symbol(r'\sum_{j}( ' + str(sp.simplify(symb_g)) + ')')
    if message_passing:
        symb_h_in = [sp.Symbol('x_i'), aggr_term]
    else:
        symb_h_in = [sp.Symbol('x_i')]
        
    if include_time:
        symb_h_in += [sp.Symbol('t')]
    
    symb_h = fit_kan(
        pruned_acts,
        pruned_preacts,
        symb_xs=symb_h_in,
        model_path=f"{model_path}/h_net"
    )
    
    symb_h = symb_h[0]  # Univariate functions
    return symb_h if message_passing else symb_h + aggr_term  


def fit_black_box(cached_input, cached_output, symb_xs, pysr_model = None, sample_size=-1):
    in_dim = cached_input.size(1)
    out_dim = cached_output.size(1)

    x = cached_input.detach().numpy().reshape(-1, in_dim)
    y = cached_output.detach().numpy().reshape(-1, out_dim)

    top_5_eq = fit_acts_pysr(x, y, pysr_model=pysr_model, sample_size=sample_size)
    symb_func = sp.sympify(top_5_eq["sympy_format"].iloc[0])

    subs_dict = {sp.Symbol(f'x{i}'): symb_xs[i] for i in range(len(symb_xs))}

    symb_func = symb_func.subs(subs_dict)

    return symb_func, top_5_eq[["complexity", "loss", "score", "sympy_format"]]



def fit_mpnn(model_path, device='cpu', pysr_model = None, sample_size=-1, message_passing=True, include_time=False):
    # G_Net
    cached_input = torch.load(f'{model_path}/g_net/cached_data/cached_input', weights_only=False, map_location=torch.device(device))
    cached_output = torch.load(f'{model_path}/g_net/cached_data/cached_output', weights_only=False, map_location=torch.device(device))
    symb_g, top_5_eqs_g = fit_black_box(
        cached_input, cached_output, 
        symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')], 
        pysr_model=pysr_model,
        sample_size=sample_size
    )
    top_5_eqs_g.to_csv(f"{model_path}/top_5_equations_g.csv")
    
    # H_Net
    cached_input = torch.load(f'{model_path}/h_net/cached_data/cached_input', weights_only=False, map_location=torch.device(device))
    cached_output = torch.load(f'{model_path}/h_net/cached_data/cached_output', weights_only=False, map_location=torch.device(device))

    aggr_term = sp.Symbol(r'\sum_{j}( ' + str(symb_g) + ')')
    
    if message_passing:
        symb_h_in = [sp.Symbol('x_i'), aggr_term]
    else:
        symb_h_in = [sp.Symbol('x_i')]
        
    if include_time:
        symb_h_in += [sp.Symbol('t')]
    
    symb_h, top_5_eqs_h = fit_black_box(
        cached_input, 
        cached_output, 
        symb_xs=symb_h_in, 
        pysr_model=pysr_model,
        sample_size=sample_size
    )
    top_5_eqs_h.to_csv(f"{model_path}/top_5_equations_h.csv")

    return symb_h if message_passing else symb_h + aggr_term


def fit_black_box_from_kan(
    model_path, 
    n_g_hidden_layers, 
    n_h_hidden_layers, 
    device='cpu', 
    theta=0.1, 
    pysr_model = None, 
    sample_size=-1,
    message_passing=True,
    include_time=False
    ):
    #G_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/g_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)

    input = pruned_preacts[0]
    output = pruned_acts[-1].sum(dim=2)

    symb_g, top_5_eqs_g = fit_black_box(
        input, 
        output, 
        symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')], 
        pysr_model=pysr_model,
        sample_size=sample_size
    )
    
    save_path = f"{model_path}/black-box"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    top_5_eqs_g.to_csv(f"{save_path}/top_5_equations_g.csv")

    #H_Net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_h_hidden_layers, model_path=f'{model_path}/h_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)

    input = pruned_preacts[0]
    output = pruned_acts[-1].sum(dim=2)

    aggr_term = sp.Symbol(r'\sum_{j}( ' + str(symb_g) + ')')
    
    if message_passing:
        symb_h_in = [sp.Symbol('x_i'), aggr_term]
    else:
        symb_h_in = [sp.Symbol('x_i')]
    
    if include_time:
        symb_h_in += [sp.Symbol('t')]

    symb_h, top_5_eqs_h = fit_black_box(
        input, 
        output, 
        symb_xs=symb_h_in, 
        pysr_model=pysr_model,
        sample_size=sample_size
    )
    top_5_eqs_h.to_csv(f"{save_path}/top_5_equations_h.csv")

    return symb_h if message_passing else symb_h + aggr_term


def quantise(expr, quantise_to=0.01):
    if isinstance(expr, sympy.Float):
        return expr.func(round(float(expr) / quantise_to) * quantise_to)
    elif isinstance(expr, (sympy.Symbol, sympy.Integer)):
        name = str(expr)
        match = re.match(r'\\sum_\{[^}]*\}\((.*)\)', name)
        if match:
            inner_expr_str = match.group(1)
            try:
                # Convert inner string to sympy expression
                inner_expr = sympy.sympify(inner_expr_str)
                # Quantise inner expression
                quantised_inner = quantise(inner_expr, quantise_to)
                # Rebuild symbol name
                new_name = re.sub(r'\(.*\)', f'({quantised_inner})', name)
                return sympy.Symbol(new_name)
            except (sympy.SympifyError, SyntaxError):
                return expr  # If parsing fails, return the symbol as-is
        else:
            return expr
    else:
        return expr.func(*[quantise(arg, quantise_to) for arg in expr.args])
    
                
def top_down_fitting(
        symb_in,
        pruned_acts,
        pruned_preacts,
        saving_path,
        pysr_model=None,
        sample_size=-1,
        neuron_level = False
):
    
    func_hierarchy = defaultdict(dict)
    for index, l_post in enumerate(reversed(range(len(pruned_acts)))): # Starting from last layer
        black_box_input = pruned_preacts[0]
        symb_xs=symb_in
        for l_prev in range(l_post, len(pruned_acts)):
            symb_neurons = []
            for j in range(pruned_acts[l_prev].shape[1]):
                if neuron_level:
                    symb_neuron_j,_ = fit_black_box(
                        black_box_input,
                        pruned_acts[l_prev].sum(dim=2)[:, j].unsqueeze(-1),
                        symb_xs=symb_xs,
                        pysr_model=pysr_model,
                        sample_size=sample_size
                )
                else:
                    symb_neuron_j = 0
                    for i in range(pruned_acts[l_prev].shape[2]):
                        input_spline = black_box_input if (l_post == l_prev and l_prev > 0)  else black_box_input[:, i].unsqueeze(-1)
                        symb_xs_spline = symb_xs if (l_post == l_prev and l_prev > 0) else [symb_xs[i]]
                        black_box_spline, _ = fit_black_box(
                            cached_input=input_spline,
                            cached_output=pruned_acts[l_prev][:, j, i].unsqueeze(-1),
                            symb_xs=symb_xs_spline,
                            pysr_model=pysr_model,
                            sample_size=sample_size
                        )
                        symb_neuron_j += black_box_spline
                        func_hierarchy[f"f_{index}"][f"spline_{l_prev}_{j}_{i}"] = str(black_box_spline)
                        
                symb_neurons.append(symb_neuron_j)
                func_hierarchy[f"f_{index}"][f"neuron_{l_prev}_{j}"] = str(symb_neuron_j)
                
            black_box_input = pruned_acts[l_prev].sum(dim=2)
            symb_xs = symb_neurons
            
    with open(saving_path, "w") as f:
            json.dump(func_hierarchy, f)
            
    
    
                
   
def hierarchical_symb_fitting(
    model_path,
    theta=0.1,
    n_g_hidden_layers = 2,
    n_h_hidden_layers = 2,
    pysr_model = None,
    sample_size=-1,
    message_passing=False,
    include_time=False,
    neuron_level=True
):
    
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/g_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)
    file_name = 'spline_top_down.json' if not neuron_level else 'neuron_top_down.json'
    saving_path = f"{model_path}/g_net"
    os.makedirs(saving_path, exist_ok=True)
    
    
    top_down_fitting(
        symb_in=[sp.Symbol('x_i'), sp.Symbol('x_j')],
        pruned_acts=pruned_acts,
        pruned_preacts=pruned_preacts,
        pysr_model=pysr_model,
        sample_size=sample_size,
        neuron_level=neuron_level,
        saving_path=f"{saving_path}/{file_name}"
    )
    
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_h_hidden_layers, model_path=f'{model_path}/h_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)
    saving_path = f"{model_path}/h_net" 
    os.makedirs(saving_path, exist_ok=True)
    
    if message_passing:
        symb_h_in = [sp.Symbol('x_i'), sp.Symbol('AGGR')]
    else:
        symb_h_in = [sp.Symbol('x_i')]
        
    if include_time:
        symb_h_in += [sp.Symbol('t')]
    
    top_down_fitting(
        symb_in=symb_h_in,
        pruned_acts=pruned_acts,
        pruned_preacts=pruned_preacts,
        pysr_model=pysr_model,
        sample_size=sample_size,
        neuron_level=neuron_level,
        saving_path=f"{saving_path}/{file_name}"
    )
    
    
    
    
    
    
    
    # # Create symbolic feature library
# def build_symbolic_library(x_dim):
#     x_syms = [sp.Symbol(f'x{i}') for i in range(x_dim)]

#     exprs = []  # constant term
#     for i in range(x_dim):
#         exprs += [0, x_syms[i], x_syms[i]**2, sp.sin(x_syms[i]), sp.cos(x_syms[i]), sp.exp(x_syms[i]), sp.log(sp.Abs(x_syms[i]) + 1e-5)]
#     if x_dim == 2:
#         x0, x1 = x_syms
#         exprs += [x0 * x1, sp.sin(x0 - x1), sp.cos(x0 - x1), (1 - x0) * x1, (1 - x1) * x0, (x0 + x1) / 2, (x0 - x1) / 2]
    
#     return exprs


# def evaluate_library(exprs, x_np):
#     x_dim = x_np.shape[1]
#     x_syms = sp.symbols([f'x{i}' for i in range(x_dim)])  # (x0, x1, ..., xd)
    
#     funcs = [sp.lambdify(x_syms, expr, modules='numpy') for expr in exprs]

#     X_cols = []
#     for f in funcs:
#         try:
#             values = f(*x_np.T)
#             values = np.atleast_1d(values)
#             if values.shape[0] == 1:
#                 # Broadcast constant term
#                 values = np.full(x_np.shape[0], values.item())
#             X_cols.append(values)
#         except Exception as e:
#             print(f"Failed to evaluate function: {f}. Error: {e}")
#             # Fallback: column of zeros
#             X_cols.append(np.zeros(x_np.shape[0]))

#     return np.column_stack(X_cols)


# def fit_sindy_like(x, y, gamma=0.01):
#     x_np = x
#     if x_np.ndim == 1:
#         x_np = x_np[:, None]
#     y_np = y
    
#     exprs = build_symbolic_library(x_np.shape[1])
#     Theta = evaluate_library(exprs, x_np)

#     # Lasso for sparsity
#     model = Lasso(alpha=gamma, fit_intercept=False, max_iter=10000)
#     model.fit(Theta, y_np)
    
#     # Build symbolic expression
#     expr = sum(coef * term for coef, term in zip(model.coef_, exprs) if abs(coef) > 1e-2)
#     if expr == 0:
#         expr = sp.sympify(0)

#     return expr


# def fit_acts_sindy(x, y, sample_size=-1, seed=42, gamma=0.01):
#     rng = np.random.default_rng(seed)  
#     if sample_size > 0 and sample_size < len(x):
#         indices = rng.choice(len(x), sample_size, replace=False)
#         x_sampled = x[indices]
#         y_sampled = y[indices]
#     else:
#         x_sampled = x
#         y_sampled = y

#     return fit_sindy_like(x_sampled, y_sampled, gamma=gamma)