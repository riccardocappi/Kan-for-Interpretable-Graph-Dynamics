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
    
set_pytorch_seed(0)

from tsl.data.preprocessing.scalers import MinMaxScaler
import sympytorch
import sympy as sp
import copy
from post_processing import get_model
from torch.optim import LBFGS
from utils.utils import save_logs


def get_symb_model(model_type, device):
    if model_type == "GKAN":
        model_path = "./saved_models_optuna/model-real-epid-gkan/real_epid_gkan_7/0"
    
        def build_symb_gkan():
            x_i, x_j = sp.symbols('x_i x_j')

            a = 2.4682064
            b = 2.4648788 
            c = -0.0039747115

            g_symb = sympytorch.SymPyModule(expressions=[c * sp.exp(x_j)]) 
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
            symb_model = symb_model.to(device)
            
            return symb_model
        
        return model_path, build_symb_gkan
    
    elif model_type == "MPNN":
        model_path = "./saved_models_optuna/model-real-epid-mpnn/real_epid_mpnn_7/0"

        def build_symb_mpnn_to_opt():
            x_i, x_j = sp.symbols('x_i x_j')

            a = 3.7716758
            b = 1.9867662
            c = 1.2657967

            eps = 1e-6
            expr1 = sp.ln(sp.Max(sp.tan(x_i + c)**2 + 1, eps))
            expr2 = a * sp.ln(sp.Max(x_i + b, eps))

            g_symb = sympytorch.SymPyModule(expressions=[expr1])
            h_symb = sympytorch.SymPyModule(expressions=[expr2])
            
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
            symb_model = symb_model.to(device)
            
            return symb_model
        return model_path, build_symb_mpnn_to_opt
    elif model_type == "TSS":
        model_path = "./saved_models_optuna/tss/real_epid_covid"
        def build_symb_model_tss_to_opt():
            x_i, x_j = sp.symbols('x_i x_j')    

            a = 0.074
            b = 7.130
            expr1 = b * (1 / (1 + sp.exp(- (x_j - x_i))))
            expr2 = a * x_i
            
            g_symb = sympytorch.SymPyModule(expressions=[expr1])
            h_symb = sympytorch.SymPyModule(expressions=[expr2])
            
            g_symb = symb_wrapper(g_symb, is_self_interaction=False)
            h_symb = symb_wrapper(h_symb, is_self_interaction=True)

            symb_model = get_model(
                g = g_symb,
                h = h_symb,
                message_passing=False,
                include_time=False,
                integration_method='euler',
                eval=False,
                all_t=True
            )
            
            symb_model = symb_model.train()   
            symb_model = symb_model.to(device)
            
            return symb_model
    
        return model_path, build_symb_model_tss_to_opt
    elif model_type == "LLC":
        model_path = "./saved_models_optuna/model-real-epid-llc/real_epid_llc_3/0"
        def build_symb_llc_to_opt():
            x_i, x_j = sp.symbols('x_i x_j')    

            a = 3.3846776
            b = 0.99761075
            c = 1.0
            expr1 = c*((x_i - x_j) * sp.exp(- x_j))
            expr2 = a * sp.tanh(x_i + b)
            
            g_symb = sympytorch.SymPyModule(expressions=[expr1])
            h_symb = sympytorch.SymPyModule(expressions=[expr2])
            
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
            symb_model = symb_model.to(device)
            
            return symb_model
        return model_path, build_symb_llc_to_opt
    
    elif model_type == 'SW':
        model_path = "./saved_models_optuna/model-real-epid-gkan/real_epid_gkan_7/0"
        def build_symb_sw_to_opt():
            x_i, x_j = sp.symbols('x_i x_j')
            a = 0.00793399829034096
            b = 2.49162501367285
            c = 4.60957300897616
            d = 0.0179951346099805
            e = 0.867764356138973
            f = 1.41939897565132
            g = 0.0114351987094713
            
            expr1 = a*sp.tanh(b*x_i - c) - d*sp.tanh(e*x_j - f) - g
            
            h = 1.19863983320216
            i = 0.573780466363341
            j = 1.29877970942686
            k = 0.937609075806
            l = 0.786015724237153
            m = 2.90646585491406
            n = 0.1771112458638
            o = 0.0871818528414837
            p = 1.36529222873796
            q = 0.587179376084621
            r = 1.08466950319977
             
            expr2 = h*sp.tanh(i*sp.tanh(j*x_i + k) + l) - m*sp.tanh(n*x_i**3 + o*x_i**2 - p*x_i - q) + r
            
            g_symb = sympytorch.SymPyModule(expressions=[expr1])
            h_symb = sympytorch.SymPyModule(expressions=[expr2])
            
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
            symb_model = symb_model.to(device)
            
            return symb_model

        return model_path, build_symb_sw_to_opt
            
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
    val_perc=0.1
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
    valid_end = int(T * (tr_perc + val_perc))

    results_df = pd.DataFrame()
    lr_grid = [1e-2, 1e-3]
    epochs_grid = [50, 100]
    logs_file = f"{model_path}/logs_fine_tuning_{data.root}.txt"

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

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                    
                    y_pred = get_predictions(data_0, model, node)
                    y_true = data[0].y[:, node, :].to(device)
                    if scaler is not None:
                        y_true = scaler.transform(y_true.cpu()).to(device)

                    loss = loss_fn(y_pred[:tr_end], y_true[:tr_end])
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        model.eval()
                        y_pred_val = get_predictions(data_0, model, node)[tr_end:valid_end]
                        val_loss = loss_fn(y_pred_val, y_true[tr_end:valid_end]).item()

                    if val_loss < best_val_loss_config:
                        best_val_loss_config = val_loss
                        best_epoch = epoch
                        best_model_state_config = copy.deepcopy(model.state_dict())

                    elif epoch - best_epoch > patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    if epoch % log == 0:
                        mes = f"Epoch {epoch}, train loss: {loss.item():.4f}, val loss: {val_loss:.4f}"
                        print(mes)
                        save_logs(
                            file_name=logs_file, 
                            log_message=mes,
                            save_updates=True
                        )

                # Save best model across grid
                if best_val_loss_config < best_val_loss:
                    best_val_loss = best_val_loss_config
                    best_model_state = copy.deepcopy(best_model_state_config)
                    best_config = (lr, max_epochs)

        print(f"Best config for {country_name}: lr={best_config[0]}, epochs={best_config[1]}")
        # Load best model
        final_model = build_symb_model().to(device)
        final_model.load_state_dict(best_model_state)

        h_net = final_model.conv.model.h_net.symb
        g_net = final_model.conv.model.g_net.symb

        # Handle empty parameter lists safely
        h_params = list(h_net.parameters())
        g_params = list(g_net.parameters())

        if h_params:
            self_int_coeffs = torch.cat([p.detach().cpu().flatten() for p in h_params]).numpy()
        else:
            self_int_coeffs = np.array([])

        if g_params:
            pairwise_int_coeffs = torch.cat([p.detach().cpu().flatten() for p in g_params]).numpy()
        else:
            pairwise_int_coeffs = np.array([])

        # Concatenate safely even if one or both are empty
        coeffs = np.concatenate([self_int_coeffs, pairwise_int_coeffs])
        results_df[country_name] = coeffs
        
        # print(f"Inferred coeffs for {country_name}: {coeffs}")
        save_logs(
            file_name=logs_file, 
            log_message=f"Inferred coeffs for {country_name}: {coeffs}",
            save_updates=True
        )
        

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
    parser.add_argument('--scale', action='store_true', help='Whether to scale the input data')
    
    args = parser.parse_args() 
    
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
        
    model_path, build_symb_model = get_symb_model(args.model_type, device=args.device)
    
    use_scale = args.scale
    scaler = None
    if use_scale:
        print("!!!SCALING!!!")
        scaler = get_scaler(data = real_epid_data, tr_perc=0.8)
    
    fit_param_per_country_gd(
        data=real_epid_data,
        countries_dict=countries_dict,
        model_path=model_path,
        build_symb_model=build_symb_model,
        device=args.device,
        patience=30,
        save_file=args.save_file,
        scaler=scaler,
        tr_perc=0.8,
        val_perc=0.1
    )