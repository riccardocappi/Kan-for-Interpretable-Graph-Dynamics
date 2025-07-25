import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np
import argparse

def set_pytorch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

from tsl.data.preprocessing.scalers import MinMaxScaler
import torch
import sympytorch
import sympy as sp
import copy
import numpy as np
from post_processing import get_model
from torch.optim import LBFGS


def get_symb_model(model_type):
    if model_type == "GKAN":
        model_path = "./saved_models_optuna/model-real-epid-gkan/real_epid_gkan_7/0"
    
        def build_symb_gkan():
            x_i, x_j = sp.symbols('x_i x_j')

            a = 2.4682064
            b = 2.4648788 
            c = -0.0039747115

            g_symb = sympytorch.SymPyModule(expressions=[sp.Min(sp.Max(sp.exp(c * x_j), 1e-6), 1e6)])
            h_symb = sympytorch.SymPyModule(expressions=[a * x_i + b])
            
            g_symb = symb_wrapper(g_symb, is_self_interaction=False)
            h_symb = symb_wrapper(h_symb, is_self_interaction=True)

            symb_model = get_model(
                g = g_symb,
                h = h_symb,
                message_passing=False,
                include_time=False,
                integration_method='rk4',
                eval=False,
                all_t=True
            )
            
            symb_model = symb_model.train()
            symb_model = symb_model.to(args.device)
            
            return symb_model
        
        return model_path, build_symb_gkan
    elif model_type == "MPNN":
        pass
    
    else:
        raise NotImplementedError("Not supported model")
    

def get_scaler(data, tr_perc = 0.8, scale_range = (-1, 1)):
    raw_data = data.raw_data_sampled.detach().cpu().numpy() # shape (IC, T, N, 1)
    tr_len = raw_data.shape[1]
    raw_data = raw_data[0, :int(tr_perc*tr_len), :, :]
    scaler = MinMaxScaler(out_range=scale_range)
    scaler.fit(raw_data.flatten())
    
    return scaler

import pandas as pd
from typing import Dict
from datasets.RealEpidemics import RealEpidemics
import json

class symb_wrapper(torch.nn.Module):
    def __init__(self, symb, is_self_interaction = True):
        super().__init__()
        self.symb = symb
        self.is_self_interaction = is_self_interaction
    
    def forward(self, x):
        if self.is_self_interaction:
            return self.symb(x_i=x[:, 0])
        else:
            return self.symb(x_i=x[:, 0], x_j=x[:, 1])


def fit_param_per_country_gd(
    data, 
    countries_dict: Dict[str, int], 
    model_path, 
    build_symb_model, 
    loss_fn=torch.nn.L1Loss(), 
    device='cuda:0', 
    patience=10, 
    log=10, 
    save_file="inferred_coeff.csv", 
    scaler=None, 
    tr_perc=0.8,
    optimizer_type="lbfgs"  # 'lbfgs' or 'adam'
):  
    def get_predictions(data_0, model, node_idx):
        try:
            y_pred = model(data_0)
        except AssertionError:
            y_pred = torch.zeros_like(data[0].y, device=data[0].y.device)
        return torch.nan_to_num(y_pred[:, node_idx, :], nan=-1.0)

    N = data[0].x.shape[1]
    T = data[0].y.shape[0]
    tr_end = int(T * tr_perc)

    results_df = pd.DataFrame()
    lr_grid = [1e-2, 1e-3]
    epochs_grid = [50, 100]

    for country_name, node in countries_dict.items():
        print(f"\nProcessing country {country_name}")
        best_model_state = None
        best_val_loss = float('inf')
        best_config = None

        for lr in lr_grid:
            for max_epochs in epochs_grid:
                print(f"Testing config: lr={lr}, epochs={max_epochs}")
                model = build_symb_model().to(device)
                for param in model.parameters():
                    if param.data.abs().max() > 1e-5:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)

                if optimizer_type.lower() == "lbfgs":
                    optimizer = LBFGS(model.parameters(), lr=lr, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-32, tolerance_change=1e-32)
                elif optimizer_type.lower() == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                else:
                    raise ValueError("Unsupported optimizer type. Use 'lbfgs' or 'adam'.")

                best_val_loss_config = float('inf')
                best_epoch = 0
                model.train()
                data_0 = data[0]

                if scaler is not None:
                    tmp = scaler.transform(data_0.x)
                    data_0 = data_0.clone()
                    data_0.x = tmp

                data_0 = data_0.to(device)

                for epoch in range(max_epochs):
                    model.train()
                    optimizer.zero_grad()
                    
                    if optimizer_type == "lbfgs":
                        def closure():
                            optimizer.zero_grad()
                            y_pred_cl = get_predictions(data_0, model, node)
                            y_true_cl = data[0].y[:, node, :].to(device)
                            if scaler is not None:
                                y_true_cl = scaler.transform(y_true_cl.cpu()).to(device)
                            loss_cl = loss_fn(y_pred_cl[:tr_end], y_true_cl[:tr_end])
                            loss_cl.backward()
                            return loss_cl
                        loss = optimizer.step(closure)
                    else:
                        y_pred = get_predictions(data_0, model, node)
                        y_true = data[0].y[:, node, :].to(device)
                        if scaler is not None:
                            y_true = scaler.transform(y_true.cpu()).to(device)

                        loss = loss_fn(y_pred[:tr_end], y_true[:tr_end])
                        loss.backward()
                        optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        y_pred_val = get_predictions(data_0, model, node)[tr_end:]
                        val_loss = loss_fn(y_pred_val, y_true[tr_end:]).item()

                    if val_loss < best_val_loss_config:
                        best_val_loss_config = val_loss
                        best_epoch = epoch
                        best_model_state_config = copy.deepcopy(model.state_dict())

                    elif epoch - best_epoch > patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    if epoch % log == 0:
                        print(f"Epoch {epoch}, train loss: {loss.item():.4f}, val loss: {val_loss:.4f}")

                # Save best model across grid
                if best_val_loss_config < best_val_loss:
                    best_val_loss = best_val_loss_config
                    best_model_state = copy.deepcopy(best_model_state_config)
                    best_config = (lr, max_epochs)

        print(f"Best config for {country_name}: lr={best_config[0]}, epochs={best_config[1]}")
        # Load best model
        final_model = build_symb_model().to(device)
        final_model.load_state_dict(best_model_state)

        h_net = final_model.conv.model.h_net
        g_net = final_model.conv.model.g_net
        self_int_coeffs = torch.cat([p.detach().cpu().flatten() for p in h_net.parameters()]).numpy()
        pairwise_int_coeffs = torch.cat([p.detach().cpu().flatten() for p in g_net.parameters()]).numpy()
        coeffs = np.concatenate([self_int_coeffs, pairwise_int_coeffs])
        results_df[country_name] = coeffs
        print(f"Inferred coeffs for {country_name}: {coeffs}")

    results_df.to_csv(f"{model_path}/{save_file}")
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run symbolic GKAN epidemic model.")
    parser.add_argument('--root', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--infection_data', type=str, default='./data/RealEpidemics/infected_numbers_covid.csv',
                        help='CSV file with infection numbers')
    parser.add_argument('--inf_threshold', type=int, default=500,
                        help='Minimum number of infections to include a country')
    parser.add_argument('--device', type=str, required=True, help='CUDA device to use (e.g., cuda:0)')
    parser.add_argument('--save_file', type=str, required=True, help='File name to save the results')
    parser.add_argument("--model_type", type=str, default="GKAN", help="Can be GKAN or MPNN")
    
    args = parser.parse_args() 
    
    set_pytorch_seed(0)
    
    real_epid_data = RealEpidemics(
        root=args.root,
        name='RealEpid',
        predict_deriv=False,
        history=1,
        horizon=44,
        scale=False,
        infection_data=args.infection_data,
        inf_threshold=args.inf_threshold
    )
    
    with open(f"{args.root}/RealEpid/countries_dict.json", 'r') as f:
        countries_dict = json.load(f)
        
    model_path, build_symb_gkan = get_symb_model(args.model_type)
    
    scaler = get_scaler(data = real_epid_data, tr_perc=0.8)
    
    fit_param_per_country_gd(
        data=real_epid_data,
        countries_dict=countries_dict,
        model_path=model_path,
        build_symb_model=build_symb_gkan,
        device=args.device,
        patience=30,
        save_file=args.save_file,
        scaler=scaler,
        tr_perc=0.8,
        optimizer_type="adam"
    )