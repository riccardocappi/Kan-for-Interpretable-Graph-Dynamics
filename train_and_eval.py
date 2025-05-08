import torch
from torch.optim import LBFGS
import os
from models.utils.ODEBlock import ODEBlock
from utils.utils import save_logs
from collections import defaultdict
from torch.utils.data import DataLoader
import copy
from datasets.SpatioTemporalGraph import SpatioTemporalGraph
import torch.nn.functional as F


def eval_model(model:ODEBlock, valid_data, criterion, scaler = None, inverse_scale = True):
    """
    Evaluates the model
    """
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for snapshot in valid_data:
            if not torch.any(snapshot.mask):
                continue 
            
            if scaler is not None:
                snapshot.x = scaler.transform(snapshot.x).squeeze(0)
                snapshot.y = scaler.transform(snapshot.y).squeeze(0)
            
            y_true.append(snapshot.y[snapshot.mask])
            y_pred.append(model(snapshot=snapshot))
        
        if len(y_pred) == 0 and len(y_true) == 0:
            return 0.0
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        
        if scaler is not None and inverse_scale:
            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y_true)
        
        loss = criterion(y_pred, y_true)
            
    return loss.item()


def fit(model:ODEBlock,
        training_set:SpatioTemporalGraph,
        valid_set:SpatioTemporalGraph, 
        test_set:SpatioTemporalGraph,
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
        scaler = None
        ):
    """
    Training process
    
    Args:
        - training_set : Training set (by default is the first 80% of the time series)
        - valid_set : Validation set (by default is the last 20% of the time series)
        - epochs : Number of epochs
        - patience : Patience hyper-parameter for early-stopping
        - lr : Learning rate
        - lmbd : Hyper-parameter for regularization loss
        - log : How often to save logs to file
        - log_file_name : Name of the logs file
        - criterion : Loss function
        - opt : Optimizer
        - save_updates : Whether to save logs during training or not
        - batch_size : Sliding window size
        - stride : Stride of the sliding window Data Loader
    """
    
    train_size = len(training_set)
    batch_size_train = train_size if batch_size == -1 else batch_size
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    collate_fn = lambda samples_list: samples_list
    train_loader = DataLoader(training_set, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
    
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
        y_pred = []
        y_true = []
        for snapshot in batch_data:
            if not torch.any(snapshot.mask):
                continue 
            
            if scaler is not None:
                snapshot.x = scaler.transform(snapshot.x).squeeze(0)
                snapshot.y = scaler.transform(snapshot.y).squeeze(0)
            
            # snapshot.x = moving_average(snapshot.x, window_size=3)
            # snapshot.y = moving_average(snapshot.y, window_size=3)
            
            y_true.append(snapshot.y[snapshot.mask])
            y_pred.append(model(snapshot=snapshot))
        
        if len(y_pred) == 0 and len(y_true) == 0:
            return 0.0
            
        y_pred = torch.cat(y_pred, dim=0) 
        y_true = torch.cat(y_true, dim=0)
        
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
            valid_data=valid_set,
            criterion=criterion,
            scaler=scaler, 
            inverse_scale=False
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
    
    # Compute test loss
    test_loss = eval_model(
        model=model,
        valid_data=test_set,
        criterion=criterion,
        scaler=scaler,
        inverse_scale=True
    )
    log_message = f"Test loss: {test_loss}"
    save_logs(logs_file_path, log_message, save_updates)
    results['test_loss'] = test_loss     
    
    return results 


def moving_average(tensor, window_size):
    # Reshape to (1, N, T) so we can use 1D convolution
    tensor = tensor.permute(1, 2, 0)  # (N, 1, T)

    # Define the convolution kernel
    kernel = torch.ones(1, 1, window_size, device=tensor.device) / window_size

    # Apply 1D convolution with padding to keep size (T)
    padding = window_size // 2
    averaged = F.conv1d(tensor, kernel, padding=padding, groups=1)

    # Handle even window size (to keep shape aligned)
    if window_size % 2 == 0:
        averaged = averaged[:, :, :-1]

    # Restore shape to (T, N, 1)
    averaged = averaged.permute(2, 0, 1)  # (T, N, 1)

    return averaged