import torch
from torch.optim import LBFGS
import os
from models.utils.ODEBlock import ODEBlock
from utils.utils import save_logs
from collections import defaultdict
from torch.utils.data import DataLoader
import copy
from datasets.SpatioTemporalGraph import SpatioTemporalGraph
from torch_geometric.loader import DataLoader as PyGDataLoader


def get_data_loader(
    training_set:SpatioTemporalGraph,
    valid_set:SpatioTemporalGraph,
    batch_size_train=-1,
    pred_deriv=False
):
    if pred_deriv:
        train_loader = PyGDataLoader(training_set, batch_size=batch_size_train, shuffle=True)
        valid_loader = PyGDataLoader(valid_set, batch_size=len(valid_set), shuffle=False)
    else:
        collate_fn = lambda samples_list: samples_list
        train_loader = DataLoader(training_set, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=len(valid_set), shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader
        

def get_predictions(model:ODEBlock, batch_data, scaler=None, pred_deriv=False):
    y_pred, y_true = None, None
    if pred_deriv:
        if scaler is not None:
            batch_data.x = scaler.transform(batch_data.x)
        y_pred = model(batch_data)
        y_true = batch_data.y
    else:
        y_pred = []
        y_true = []
        for snapshot in batch_data:
            if scaler is not None:
                snapshot.x = scaler.transform(snapshot.x)
                snapshot.y = scaler.transform(snapshot.y)
                
            y_true.append(snapshot.y[snapshot.backprop_idx] if model.training else snapshot.y)
            y_pred.append(model(snapshot=snapshot))
            
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
    return y_pred, y_true


def eval_model(model:ODEBlock, valid_loader, criterion, scaler = None, inverse_scale = True, pred_deriv=False):
    """
    Evaluates the model
    """
    model.eval()
    y_pred= []
    y_true = []
    with torch.no_grad():
        for batch_data in valid_loader:
            y_pred_batch, y_true_batch = get_predictions(model, batch_data, scaler=scaler, pred_deriv=pred_deriv)
            y_pred.append(y_pred_batch)
            y_true.append(y_true_batch)
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        if scaler is not None and inverse_scale and not pred_deriv:
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y_true)
        
        loss = criterion(y_pred, y_true)
            
    return loss.item()


def fit(model:ODEBlock,
        training_set:SpatioTemporalGraph,
        valid_set:SpatioTemporalGraph, 
        epochs=50,
        patience=30,
        lr = 0.001,
        lmbd=0.,
        log=10,
        log_file_name='logs.txt',
        criterion = torch.nn.MSELoss(),
        opt='Adam',
        save_updates=True,
        batch_size=-1,
        scaler = None,
        pred_deriv=False
        ):
    """
    Training process
    """
    
    train_size = len(training_set)
    batch_size_train = train_size if batch_size == -1 else batch_size
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    train_loader, valid_loader = get_data_loader(
        training_set=training_set,
        valid_set=valid_set,
        batch_size_train=batch_size_train,
        pred_deriv=pred_deriv
    )
    
    # collate_fn = lambda samples_list: samples_list
    # train_loader = DataLoader(training_set, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'LBFGS':
        optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        raise Exception('Optimizer not implemented yet!')
    
    if save_updates:
        logs_folder = f'{model.model_path}/logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        logs_file_path = f'{logs_folder}/{log_file_name}'
    else:
        logs_file_path = ''
    
    results = defaultdict(list)
    reg_loss_metrics = defaultdict(float)
    
    global running_training_loss, running_tot_loss
    
    def training():
        global running_training_loss, running_tot_loss
        optimizer.zero_grad()
        
        y_pred, y_true = get_predictions(model, batch_data, scaler=scaler, pred_deriv=pred_deriv)
        training_loss = criterion(y_pred, y_true)
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
        for batch_data in train_loader:  
            if opt == 'Adam':
                _ = training()
            else:
                optimizer.step(training)
            count += 1
        
        val_loss = eval_model(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
            scaler=scaler, 
            inverse_scale=False,
            pred_deriv=pred_deriv
        )
        
        results['train_loss'].append(running_training_loss / count)
        results['validation_loss'].append(val_loss)
        results['tot_loss'].append(running_tot_loss/ count)
        
        for key, value in reg_loss_metrics.items():
            results[key].append(value / count)
        
        if epoch % log == 0:
            log_message = f"Epoch: {epoch} \t Training loss: {running_training_loss/count:0.5f} \t Val Loss: {val_loss:0.5f} \t Tot Loss: {running_tot_loss/count:0.5f}"
            save_logs(logs_file_path, log_message, save_updates)
        if val_loss < best_val_loss:    # Save best moedel state so far
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch > patience: # Early stopping
            log_message = f"Early stopping at epoch {epoch}"
            save_logs(logs_file_path, log_message, save_updates)
            break
        
    log_message = f"\nLoading best model found at epoch {best_epoch} with val loss {best_val_loss}" 
    save_logs(logs_file_path, log_message, save_updates)
    model.load_state_dict(best_model_state)
        
    return results