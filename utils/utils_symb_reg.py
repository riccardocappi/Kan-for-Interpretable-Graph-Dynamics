import numpy as np
import sympy as sp
import torch
from sklearn.linear_model import Lasso

# Create symbolic feature library
def build_symbolic_library(x_dim):
    x_syms = [sp.Symbol(f'x{i}') for i in range(x_dim)]

    exprs = [1]  # constant term
    for i in range(x_dim):
        exprs += [x_syms[i], x_syms[i]**2, sp.sin(x_syms[i]), sp.cos(x_syms[i]), sp.exp(x_syms[i]), sp.log(sp.Abs(x_syms[i]) + 1e-5)]
    if x_dim == 2:
        x0, x1 = x_syms
        exprs += [x0 * x1, sp.sin(x0 - x1), sp.cos(x0 - x1)]
    
    return exprs

def evaluate_library(exprs, x_np):
    x_dim = x_np.shape[1]
    funcs = [sp.lambdify([sp.symbols([f'x{i}' for i in range(x_dim)])], expr, modules='numpy') for expr in exprs]
    X = np.column_stack([f(*x_np.T) for f in funcs])
    return X

def fit_sindy_like(x, y, alpha=0.01):
    x_np = x.detach().cpu().numpy()
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    y_np = y.detach().cpu().numpy()
    
    exprs = build_symbolic_library(x_np.shape[1])
    Theta = evaluate_library(exprs, x_np)

    # Lasso for sparsity
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(Theta, y_np)
    
    # Build symbolic expression
    expr = sum(coef * term for coef, term in zip(model.coef_, exprs) if abs(coef) > 1e-4)
    if expr == 0:
        expr = sp.sympify(0)

    return expr


def fit_acts_sindy(x, y, sample_size=-1, seed=42, alpha=0.01):
    rng = np.random.default_rng(seed)  
    if sample_size > 0 and sample_size < len(x):
        indices = rng.choice(len(x), sample_size, replace=False)
        x_sampled = x[indices]
        y_sampled = y[indices]
    else:
        x_sampled = x
        y_sampled = y

    return fit_sindy_like(x_sampled, y_sampled, alpha=alpha)
