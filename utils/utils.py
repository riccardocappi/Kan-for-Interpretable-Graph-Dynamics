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
from sklearn.metrics import r2_score
import warnings
import sympy
import re
warnings.filterwarnings("ignore", category=FutureWarning)


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
    'neg': lambda x: -x,
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
    'neg': lambda x: -x,
    '0': lambda x: 0 * x,
}

FUNC_COMPLEXITY = {
    '0': 0,
    'x': 1,
    'neg': 1,
    'x^2': 2,
    'x^3': 3,
    'abs': 2,
    'sin': 4,
    'cos': 4,
    'tan': 4,
    'tanh': 4,
    'exp': 5,
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
                plt.plot(preact_sorted[:, i].cpu().detach().numpy(), out.cpu().detach().numpy())
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
    
    Args:
        - layers : KAN layers
        - symb_functions_file : Path to the file in which are saved the fitted symbolic functions
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


def penalized_loss(y_true, y_pred, func_name, alpha=0.1):
    r2 = r2_score(y_true, y_pred)
    complexity = FUNC_COMPLEXITY.get(func_name, 5)  # default penalty if unknown
    penalty = alpha * complexity
    return r2 - penalty


def fit_params_scipy(x, y, func, func_name, device='cuda', alpha=0.1):    
    func_optim = lambda x, a, b, c, d: c * func(a*x + b) + d
    try:
        params, _ = curve_fit(func_optim, x, y, p0=[1., 0., 1., 0.], nan_policy='omit')
    except RuntimeError:
        return 1e-8, torch.tensor(np.array([1.,0.,1.,0.]), device=device, dtype=torch.float32)
    post_fun = params[2] * func(params[0]*x + params[1]) + params[3]
        
    r2 = penalized_loss(y, post_fun, func_name, alpha=alpha)
    return r2, torch.tensor(params, device=device, dtype=torch.float32)


def fit_acts_scipy(x, y, sample_size = -1, device='cuda', seed=42, alpha=0.1):
    rng = np.random.default_rng(seed)  
    if sample_size > 0 and sample_size < len(x):
        indices = rng.choice(len(x), sample_size, replace=False)
        x_sampled = x[indices]
        y_sampled = y[indices]
    else:
        x_sampled = x
        y_sampled = y
    
    scores = []
    for name, func in SYMBOLIC_LIB_NUMPY.items():
        r2, params = fit_params_scipy(x_sampled, y_sampled, func, name, device=device, alpha=alpha)
        scores.append((name, r2, params))
    
    func_name, best_r2, best_params = max(scores, key=lambda x: x[1])
    if best_r2 == 1e-8: # if the fitting fails, we set the zero function
        func_name = '0'
    
    best_fun_sympy = SYMBOLIC_LIB_SYMPY[func_name]
    x_symb = sp.Symbol('x0')    # This symbol must be x0 in order to work with the rest of the code
    best_fun_sympy = best_params[2] * best_fun_sympy(best_params[0]*x_symb + best_params[1]) + best_params[3]
    
    return best_fun_sympy
    

def fit_layer(cached_act, cached_preact, symb_xs, device='cuda', sample_size=-1, alpha=0.1):
    symb_layer_acts = []
    in_dim = cached_act.shape[2]
    out_dim = cached_act.shape[1]
    symbolic_functions = defaultdict(dict)

    for j in range(out_dim):
        symb_out = 0
        for i in range(in_dim):
            x = cached_preact[:, i].reshape(-1)
            y = cached_act[:, j, i].reshape(-1)
            
            symb_func = fit_acts_scipy(x, y, sample_size=sample_size, device=device, alpha=alpha)
            symbolic_functions[j][i] = str(symb_func)
            
            symb_func = symb_func.subs(sp.Symbol('x0'), symb_xs[i])
            symb_out += symb_func

        symb_layer_acts.append(symb_out)

    return symb_layer_acts, symbolic_functions


def fit_kan(kan_acts, kan_preacts, symb_xs, sample_size=-1, device='cuda', model_path='./models', alpha=0.1):
    n_layers = len(kan_acts)
    all_functions = {}
    
    for l in range(n_layers):
        acts = kan_acts[l].cpu().detach().numpy()
        preacts = kan_preacts[l].cpu().detach().numpy()

        symb_xs, symb_functions = fit_layer(
            cached_act=acts,
            cached_preact=preacts,
            symb_xs=symb_xs,
            device=device,
            sample_size=sample_size,
            alpha=alpha
        )
        
        all_functions[l] = symb_functions
        
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


def fit_model(n_h_hidden_layers, n_g_hidden_layers, model_path, theta=0.1, device = 'cuda', sample_size=-1, message_passing=True, include_time=False, alpha=0.1):
    # G_net
    cache_acts, cache_preacts = get_kan_arch(n_layers=n_g_hidden_layers, model_path=f'{model_path}/g_net')
    pruned_acts, pruned_preacts = pruning(cache_acts, cache_preacts, theta=theta)    
    symb_g = fit_kan(
        pruned_acts,
        pruned_preacts,
        symb_xs=[sp.Symbol('x_i'), sp.Symbol('x_j')],
        sample_size=sample_size,
        device=device,
        model_path=f"{model_path}/g_net",
        alpha=alpha
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
        sample_size=sample_size,
        device=device,
        model_path=f"{model_path}/h_net",
        alpha=alpha
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