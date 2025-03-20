import torch
from torch.optim import LBFGS
import os
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from models.utils.NetWrapper import NetWrapper
from utils.utils import save_logs
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets.SlidingWindowSampler import SlidingWindowSampler
import copy


def call_ODE(model, y0, t):
    """
    Integrates the model using dopri5 numerical integrator
    
    Args:
        -model : model to be integrated
        -y0 : initial condition
        -t : time steps in which the ODE is evaluated
    """
    return odeint(
        model, 
        y0, 
        t, 
        method='dopri5',
        atol=1e-6,
        rtol=1e-3,
        adjoint_options=dict(norm="seminorm")
    )

def eval_model(model, data, t, criterion, t_f_train, n_iter=1):
    """
    Integrates the model starting from the very first y0 until the end of the time series, and computes the loss only
    with respect to validation set.
    
    Args:
        -data : The whole time series
        -t : Time steps in which the ODE is evaluated
        -criterion : Loss function
        -t_f_train : Last index of training set
        -n_iter : Number of initial conditions
    """
    model.eval()
    y_pred = []
    with torch.no_grad():
        for k in range(n_iter):
            y_true_valid = data[k]
            t_valid = t[k]
            y0 = y_true_valid[0]
            y_pred.append(call_ODE(model, y0, t_valid)[t_f_train:])
            
        y_pred = torch.stack(y_pred, dim=0)
        loss = criterion(y_pred, data[:, t_f_train:, :, :])
    return loss.item()
            
    
    

def fit(model:NetWrapper,
        training_set,
        valid_set, 
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
        t_f_train=240,
        stride = 1
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
        - n_iter : Number of initial conditions
        - batch_size : Sliding window size
        - t_f_train : Last index of training set
        - stride : Stride of the sliding window Data Loader
    """
    
    train_size = len(training_set)
    batch_size_train = train_size if batch_size == -1 else batch_size
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    sampler = SlidingWindowSampler(training_set, batch_size_train, stride, shuffle=True)
    train_loader = DataLoader(training_set, batch_sampler=sampler)
    
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif opt == 'LBFGS':
        optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        raise Exception('Optimizer not implemented yet!')
    
    if save_updates:
        logs_folder = f'{model.model.model_path}/logs'
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
        for k in range(n_iter):
            y0 = batch_data[:, k, :, :][0]
            t_eval = norm_batch_times[:, k]
            t = torch.tensor([t_eval[0], t_eval[1], t_eval[-1]], dtype=y0.dtype).to(torch.device(y0.device))
            y_pred.append(call_ODE(model, y0, t)[1:])
        
        y_pred = torch.stack(y_pred, dim=1) # Shape (2, n_iter, n_nodes, in_dim)
        
        y_pred_flatten = y_pred.view(-1, 1)
        y_true_flatten = batch_data[[1, -1], :, :, :].view(-1, 1)
        
        training_loss = criterion(y_pred_flatten, y_true_flatten)
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
            norm_batch_times = batch_times - min_values # For each window, the integrator must start from time step 0
            if opt == 'Adam':
                _ = training()
            else:
                optimizer.step(training)
            count += 1
        
        val_loss = eval_model(
            model, 
            valid_set.data, 
            valid_set.time, 
            criterion=criterion, 
            n_iter=n_iter,
            t_f_train=t_f_train
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