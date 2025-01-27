import torch
from torch.optim import LBFGS
import os
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
from models.utils.NetWrapper import NetWrapper
from utils.utils import save_logs
import json
from collections import defaultdict
from torch.utils.data import DataLoader


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
        training_set,
        valid_set, 
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
        n_iter = 1,
        batch_size=-1,
        t_f_train=240
        ):
    
    torch.manual_seed(seed)
    train_size = len(training_set)
    batch_size_train = train_size if batch_size == -1 else batch_size
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    train_loader = DataLoader(training_set, batch_size=batch_size_train, shuffle=False)
    
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
            y0 = batch_data[:, k, :, :][0]
            t_eval = norm_batch_times[:, k]
            y_pred.append(odeint(model, y0, t_eval, method='dopri5')[1:])
        
        y_pred = torch.stack(y_pred, dim=1)
        training_loss = criterion(y_pred, batch_data[1:, :, :, :])
        
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
        count = 0
        reg_loss_metrics.clear()
        for batch_data, batch_times in train_loader:  
            min_values = batch_times.min(dim=0, keepdim=True).values
            norm_batch_times = batch_times - min_values
            if opt == 'Adam':
                _ = training()
            else:
                optimizer.step(training)
            count += 1
        
        val_loss = eval_model(model, 
                              valid_set.data, 
                              valid_set.time, 
                              criterion=criterion, 
                              t_f_train=t_f_train,
                              n_iter=n_iter
                              )
        results['train_loss'].append(running_training_loss / count)
        results['validation_loss'].append(val_loss)
        results['tot_loss'].append(running_tot_loss/ count)
        
        for key, value in reg_loss_metrics.items():
            results[key].append(value / count)
        
        if epoch % log == 0:
            log_message = f"Epoch: {epoch} \t Training loss: {running_training_loss/count:0.5f} \t Val Loss: {val_loss:0.5f} \t Tot Loss: {running_tot_loss/count:0.5f}"
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