import torch
from torch.optim import LBFGS
import os
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from models.NetWrapper import NetWrapper
from utils.utils import save_logs
import json
from collections import defaultdict


def eval_model(model, data, t, criterion, t_f_train, n_iter=1):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for k in range(n_iter):
            y_true_valid = data[k]
            t_valid = t[k]
            y0 = y_true_valid[0]
            y_pred.append(odeint(model, y0, t_valid, method='dopri5')[t_f_train:])
            
        y_pred = torch.stack(y_pred, dim=0)
        loss = criterion(y_pred, data[:, t_f_train:, :, :])
    return loss.item()
            
    
    

def fit(model:NetWrapper,
        train_data, 
        t_train, 
        valid_data, 
        t_valid, 
        seed=42,
        epochs=50,
        patience=30,
        lr = 0.001,
        lmbd=0.,
        log=10,
        log_file_name='logs.txt',
        criterion = torch.nn.MSELoss(),
        opt='Adam',
        save_updates=True,
        t_f_train = 240,
        n_iter = 1
        ):
    
    torch.manual_seed(seed)
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    # y0 = train_data[0]

    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'LBFGS':
        optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        raise Exception('Optimizer not implemented yet!')
    

    logs_folder = f'{model.model.model_path}/logs'
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    logs_file_path = f'{logs_folder}/{log_file_name}'
    
    results = defaultdict(list)
    reg_loss_metrics = defaultdict(float)
    
    global running_training_loss, running_tot_loss
    
    def training():
        global running_training_loss, running_tot_loss
        optimizer.zero_grad()
        y_pred = []
        for k in range(n_iter):
            y_true = train_data[k]
            t_eval = t_train[k]
            y0 = y_true[0]
            y_pred.append(odeint(model, y0, t_eval, method='dopri5')[1:]) # We can implement batch learning
        
        y_pred = torch.stack(y_pred, dim=0)
        training_loss = criterion(y_pred, train_data[:, 1:, :, :])
        
        running_training_loss = running_training_loss + training_loss.item()
        reg = model.regularization_loss(reg_loss_metrics)
        loss = training_loss + lmbd * reg
        running_tot_loss = running_tot_loss + loss.item()
        loss.backward()
        if opt == 'Adam':
            optimizer.step()
        return loss

    
    for epoch in range(epochs):
        model.train()
        running_training_loss = 0.
        running_tot_loss = 0.
        reg_loss_metrics.clear()
        if opt == 'Adam':
            _ = training()
        else:
            optimizer.step(training)
        
        val_loss = eval_model(model, valid_data, t_valid, criterion, t_f_train, n_iter=n_iter)
        results['train_loss'].append(running_training_loss)
        results['validation_loss'].append(val_loss)
        results['tot_loss'].append(running_tot_loss)
        
        for key, value in reg_loss_metrics.items():
            results[key].append(value)
        
        if epoch % log == 0:
            log_message = f"Epoch: {epoch} \t Training loss: {running_training_loss:0.5f} \t Val Loss: {val_loss:0.5f} \t Tot Loss: {running_tot_loss:0.5f}"
            save_logs(logs_file_path, log_message, save_updates)
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        elif epoch - best_epoch > patience:
            log_message = f"Early stopping at epoch {epoch}"
            save_logs(logs_file_path, log_message, save_updates)
            break
        
    log_message = f"\nLoading best model found at epoch {best_epoch} with val loss {best_val_loss}" 
    save_logs(logs_file_path, log_message, save_updates)
    model.load_state_dict(best_model_state)
    
    if save_updates:
        torch.save(best_model_state, f'{model.model.model_path}/best_state_dict.pth')
        with open(f"{model.model.model_path}/results.json", "w") as outfile: 
            json.dump(results, outfile)
        
    return results  