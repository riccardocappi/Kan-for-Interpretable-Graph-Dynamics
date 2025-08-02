import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import sympy as sp


f_log = lambda x, y_th: ((x_th := torch.e**(-y_th)), - y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th))
f_tan = lambda x, y_th: ((clip := x % torch.pi), (delta := torch.pi/2-torch.arctan(y_th)), - y_th/delta * (clip - torch.pi/2) * (torch.abs(clip - torch.pi/2) < delta) + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi/2) >= delta))
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(x) * (x <= x_th))

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
                 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
                 'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
                 'exp': (lambda x: torch.exp(x), lambda x: sp.exp(x), 2, f_exp),
                 'log': (lambda x: torch.log(x), lambda x: sp.log(x), 2, f_log),
                 'abs': (lambda x: torch.abs(x), lambda x: sp.Abs(x), 3, lambda x, y_th: ((), torch.abs(x))),
                 'sin': (lambda x: torch.sin(x), lambda x: sp.sin(x), 2, lambda x, y_th: ((), torch.sin(x))),
                 'cos': (lambda x: torch.cos(x), lambda x: sp.cos(x), 2, lambda x, y_th: ((), torch.cos(x))),
                 'tan': (lambda x: torch.tan(x), lambda x: sp.tan(x), 3, f_tan),
                 'tanh': (lambda x: torch.tanh(x), lambda x: sp.tanh(x), 3, lambda x, y_th: ((), torch.tanh(x))),
                 '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
    }



def fit_params(x, y, fun, a_range=(-10,10), b_range=(-10,10), grid_number=101, iteration=3, verbose=True, device='cpu'): 
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None,:,:] * x[:,None,None] + b_grid[None,:,:])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean)*(y-y_mean)[:,None,None], dim=0)**2
        denominator = torch.sum((post_fun - x_mean)**2, dim=0)*torch.sum((y - y_mean)[:,None,None]**2, dim=0)
        r2 = numerator/(denominator+1e-4)
        r2 = torch.nan_to_num(r2)
        
        
        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number
        
        
        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose==True:
                print('Best value at boundary.')
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]
            
        else:
            a_range = [a_[a_id-1], a_[a_id+1]]
            b_range = [b_[b_id-1], b_[b_id+1]]
            
    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]
    
    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:,None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def fix_symbolic(fun_name, x, y, a_range=(-10, 10), b_range=(-10, 10), verbose=False, device='cpu'):
    x_symb = sp.Symbol('x0')
    fun = SYMBOLIC_LIB[fun_name][0]
    fun_sympy = SYMBOLIC_LIB[fun_name][1]
    params, r2 = fit_params(x,y,fun, a_range=a_range, b_range=b_range, verbose=verbose, device=device)
    a, b, c, d = params[0].item(), params[1].item(), params[2].item(), params[3].item()  
    sympy_ret = c * fun_sympy(a * x_symb + b) + d
    
    return r2, sympy_ret
    
    

def suggest_symbolic(x, y, a_range=(-10, 10), b_range=(-10, 10), topk=5, r2_loss_fun=lambda x: np.log2(1+1e-5-x), c_loss_fun=lambda x: x, weight_simple = 0.3, device='cpu'):
    r2s = []
    cs = []
    sympy_funcs = []
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    symbolic_lib = SYMBOLIC_LIB
    for (name, content) in symbolic_lib.items():
        r2, fun_sympy = fix_symbolic(name, x, y, a_range=a_range, b_range=b_range, verbose=False, device=device)
        if r2 == -1e8: # zero function
            r2s.append(-1e8)
            sympy_funcs.append(sp.S(0))
        else:
            r2s.append(r2.item())
            sympy_funcs.append(fun_sympy)
        
        c = content[2]
        cs.append(c)
    
    r2s = np.array(r2s)
    cs = np.array(cs)
    r2_loss = r2_loss_fun(r2s).astype('float')
    cs_loss = c_loss_fun(cs)
    
    loss = weight_simple * cs_loss + (1-weight_simple) * r2_loss
    sorted_ids = np.argsort(loss)[:topk]
    r2s = r2s[sorted_ids][:topk]
    cs = cs[sorted_ids][:topk]
    r2_loss = r2_loss[sorted_ids][:topk]
    cs_loss = cs_loss[sorted_ids][:topk]
    loss = loss[sorted_ids][:topk]
    sympy_funcs = [sympy_funcs[i] for i in sorted_ids[:topk]]
      
    # topk = np.minimum(topk, len(symbolic_lib))
    
    # best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
    # best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
    # best_r2 = r2s[0]
    # best_c = cs[0]
    # return best_name, best_fun, best_r2, best_c
    
    return sympy_funcs