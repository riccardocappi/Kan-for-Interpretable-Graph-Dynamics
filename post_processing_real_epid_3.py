import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np

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
from models.utils.MPNN import MPNN
from models.baseline.MPNN_ODE import MPNN_ODE
import torch
import sympytorch
from torch_geometric.data import Data
import sympy as sp
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from post_processing import get_model, make_callable, plot_predictions
from torch.optim import LBFGS


def get_scaler(data, tr_perc = 0.8, scale_range = (-1, 1)):
    raw_data = data.raw_data_sampled.detach().cpu().numpy() # shape (IC, T, N, 1)
    tr_len = raw_data.shape[1]
    raw_data = raw_data[0, :int(tr_perc*tr_len), :, :]
    scaler = MinMaxScaler(out_range=scale_range)
    scaler.fit(raw_data.flatten())
    
    return scaler


  
def eval_real_epid_int(data, countries_dict, build_symb_model, inferred_coeffs, scaler=None, use_euler=False, tr_perc = 0.8):
    y_true = data[0].y.detach().cpu().numpy()
    y_pred = np.zeros_like(y_true)
    
    for country_name, node_idx in countries_dict.items():
        symb_model = build_symb_model(country_name, inferred_coeffs)
        # print(f"{country_name}")
        data_0 = data[0]
        if scaler is not None:
            tmp = scaler.transform(data[0].x)
            data_0 = data[0]
            data_0.x = tmp
        
        if use_euler:
            symb_model.integration_method = "euler"
            data_0.t_span = torch.arange(y_true.shape[0] + 1, device=data_0.x.device, dtype=data_0.t_span.dtype)
        
        try:
            pred = symb_model(data_0).detach().cpu().numpy()
        except AssertionError:
            print("Failed")
            continue
        
        if scaler is not None:
            pred = scaler.inverse_transform(pred)
        
        y_pred[:, node_idx, :] = pred[:, node_idx, :]
    
    tr_len = y_true.shape[0]
    tr_end = int(tr_perc * tr_len)
    y_true_val = y_true[tr_end:, :, :]
    y_pred_val = y_pred[tr_end:, :, :] 
    
    return y_true, y_pred, y_true_val, y_pred_val 


def eval_real_epid_journal(data, countries_dict, build_symb_model, inferred_coeffs, tr_perc = 0.8, step_size=1.0, scaler = None):
    def get_dxdt_pred(data, symb_model):
        dxdt_pred = []
        for snapshot in data:
            dxdt_pred.append(symb_model(snapshot))
        
        return torch.stack(dxdt_pred, dim=0)
    
    def sum_over_dxdt(dxdt_pred):
        out = []
        for i in range(dxdt_pred.shape[0]):
            out.append(torch.sum(step_size*dxdt_pred[0:i+1, :, :], dim=0)) 
        
        return torch.stack(out, dim=0)
        
    def integrate(out, x0):
        pred = []
        for i in range(1, out.shape[0]):
            pred.append(x0 + out[i, :, :])
        return torch.stack(pred, dim=0)
      
    x0 = data[0].x
    y_true = torch.stack([d.x for d in data[1:]], dim=0).detach().cpu().numpy()
    y_pred = np.zeros_like(y_true)
     
    for country_name, node_idx in countries_dict.items():
        symb_model = build_symb_model(country_name, inferred_coeffs)
        symb_model.predict_deriv = True
        dxdt_pred = get_dxdt_pred(data, symb_model)
        out = sum_over_dxdt(dxdt_pred)
        pred = integrate(out, x0).detach().cpu().numpy()
        y_pred[:, node_idx, :] = pred[:, node_idx, :]
    
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
    
    tr_len = y_true.shape[0]
    tr_end = int(tr_perc * tr_len)
    y_true_val = y_true[tr_end:, :, :]
    y_pred_val = y_pred[tr_end:, :, :] 
    
    return y_true, y_pred, y_true_val, y_pred_val 
        



import pandas as pd
# from torch.optim import Adam
# from scipy.optimize import minimize
from typing import Dict
from torch.utils.data import DataLoader
# import optuna

    
def fit_param_per_country_gd(
    data,
    valid_data, 
    countries_dict: Dict[str, int], 
    model_path, 
    build_symb_model, 
    loss_fn=torch.nn.L1Loss(), 
    device='cuda:0', 
    patience=30, 
    log=10, 
    save_file="inferred_coeff_2.csv", 
    scaler=None, 
    tr_perc=0.8,
    optimizer_type="adam"  # 'lbfgs' or 'adam'
):
    def get_predictions(batch_data, model, node_idx):
        y_pred = []
        y_true = []
        for snapshot in batch_data:
            snapshot = snapshot.to(device)
            if scaler is not None:
                snapshot.x = scaler.transform(snapshot.x)
                snapshot.y = scaler.transform(snapshot.y)
                
            y_true.append(snapshot.y)
            y_pred.append(model(snapshot=snapshot))
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        return y_true[:, node_idx, :], y_pred[:, node_idx, :]
    
    def eval_model(model, valid_loader, node_idx):

        model.eval()
        y_pred= []
        y_true = []
        with torch.no_grad():
            for batch_data in valid_loader:
                y_true_batch, y_pred_batch = get_predictions(batch_data, model, node_idx)
                y_pred.append(y_pred_batch)
                y_true.append(y_true_batch)
            
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
                        
            loss = loss_fn(y_pred, y_true)
                
        return loss.item()
    

    # N = data[0].x.shape[1]
    # T = data[0].y.shape[0]
    # tr_end = int(T * tr_perc)

    results_df = pd.DataFrame()
    lr_grid = [1e-2, 1e-3]
    epochs_grid = [50, 100, 150]

    for country_name, node in countries_dict.items():
        print(f"\nProcessing country {country_name}")
        best_model_state = None
        best_val_loss = float('inf')
        best_config = None

        for lr in lr_grid:
            for max_epochs in epochs_grid:
                print(f"Testing config: lr={lr}, epochs={max_epochs}")
                model = build_symb_model().to(device)
                collate_fn = lambda samples_list: samples_list
                train_loader = DataLoader(data, batch_size=len(data), shuffle=True, collate_fn=collate_fn)
                valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False, collate_fn=collate_fn)
                
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

                for epoch in range(max_epochs):
                    model.train()
                    optimizer.zero_grad()
                    count = 0
                    training_loss = 0.0
                    for batch_data in train_loader:
                        count += 1
                        if optimizer_type == "adam":
                            y_true, y_pred = get_predictions(batch_data, model, node)
        
                            loss = loss_fn(y_pred, y_true)
                            loss.backward()
                            optimizer.step()

                        else:
                            def closure():
                                optimizer.zero_grad()
                                y_true, y_pred = get_predictions(batch_data, model, node)
                                loss_cl = loss_fn(y_pred, y_true)
                                loss_cl.backward()
                                return loss_cl
                            loss = optimizer.step(closure)
                        training_loss += loss.item()
                            

                    val_loss = eval_model(
                        model,
                        valid_loader,
                        node
                    )

                    if val_loss < best_val_loss_config:
                        best_val_loss_config = val_loss
                        best_epoch = epoch
                        best_model_state_config = copy.deepcopy(model.state_dict())

                    elif epoch - best_epoch > patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                    if epoch % log == 0:
                        print(f"Epoch {epoch}, train loss: {training_loss/count:.4f}, val loss: {val_loss:.4f}")

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


if __name__ == "__main__":
    from datasets.RealEpidemics import RealEpidemics

    real_epid_data = RealEpidemics(
        root = './data_real_epid_covid_int',
        name = 'RealEpid',
        predict_deriv=False,
        history=1,
        horizon=44,
        scale=False
    )
    
    import json

    with open('./data_real_epid_covid_int/RealEpid/countries_dict.json', 'r') as f:
        countries_dict = json.load(f)
        
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
            
    model_path = "./saved_models_optuna/model-real-epid-mpnn/real_epid_mpnn_7/0"

    # import random

    def build_symb_mpnn_to_opt():
        x_i, x_j = sp.symbols('x_i x_j')

        a = 3.7716758
        b = 1.9867662
        c = 1.2657967

        eps = 1e-3
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
        symb_model = symb_model.to('cuda:1')
        
        return symb_model


    fine_tune_data = RealEpidemics(
        root = './data_real_epid_covid_ft_scaled',
        name = 'RealEpid',
        predict_deriv=False,
        history=1,
        horizon=9,
        stride=7,
        scale=True,
        scale_range=(-1, 1),
        train_perc=0.8
    )

    tr_len = len(fine_tune_data)
    tr_end = int(0.8 * tr_len)
    train_set = fine_tune_data[:tr_end]
    valid_set = fine_tune_data[tr_end:]

    fit_param_per_country_gd(
        data=train_set,
        valid_data=valid_set,
        build_symb_model=build_symb_mpnn_to_opt,
        countries_dict=countries_dict,
        model_path=model_path,
        patience=30,
        log=10,
        scaler=None,
        tr_perc=0.8,
        save_file="prova_2.csv",
        optimizer_type="adam",
        device="cuda:1"
    )

        