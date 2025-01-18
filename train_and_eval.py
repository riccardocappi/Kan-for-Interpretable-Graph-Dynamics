import torch
from torch.optim import LBFGS
import os
from torchdiffeq import odeint
from models.NetWrapper import NetWrapper
from utils.utils import save_logs
import json


def eval_model(model, data, t_eval, criterion):
    model.eval()
    y0 = data[0]
    with torch.no_grad():
        y_pred = odeint(model, y0, t_eval, method='dopri5')
        loss = criterion(y_pred[1:], data[1:])
    return loss.item()
            
    
    

def fit(model:NetWrapper,
        train_data, 
        t_train, 
        valid_data, 
        t_valid, 
        test_data, 
        t_test,
        seed=42,
        epochs=50,
        patience=30,
        lr = 0.001,
        lmbd=0.,
        log=10,
        mu_1=1.,
        mu_2 = 1.,
        log_file_name='logs.txt',
        criterion = torch.nn.MSELoss(),
        opt='Adam',
        use_orig_reg=False,
        save_updates=True
        ):
    
    torch.manual_seed(seed)
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    y0 = train_data[0]

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
    results = {
        'train_loss': [],
        'validation_loss': [],
        'tot_loss': [],
        'reg': [],
        'l1':[],
        'entropy': []
    }
    
    global running_training_loss, running_tot_loss, running_reg, running_l1, running_entropy, upd_grid
    
    def training():
        global running_training_loss, running_tot_loss, running_reg, running_l1, running_entropy, upd_grid
        optimizer.zero_grad()
        y_pred = odeint(model, y0, t_train, method='dopri5')
        training_loss = criterion(y_pred[1:], train_data[1:]) # We can implement batch learning by partitioning t_train
        running_training_loss = running_training_loss + training_loss.item()
        reg, l1, entropy = model.regularization_loss(mu_1, mu_2, use_orig_reg)
        loss = training_loss + lmbd * reg
        running_tot_loss = running_tot_loss + loss.item()
        running_reg = running_reg + reg.item()
        running_l1 = running_l1 + l1.item()
        running_entropy = running_entropy + entropy.item()
        loss.backward()
        if opt == 'Adam':
            optimizer.step()
        return loss

    
    for epoch in range(epochs):
        model.train()
        running_training_loss = 0.
        running_tot_loss = 0.
        running_reg = 0.
        running_l1 = 0.
        running_entropy = 0.
        if opt == 'Adam':
            _ = training()
        else:
            optimizer.step(training)
        
        val_loss = eval_model(model, valid_data, t_valid, criterion)
        results['train_loss'].append(running_training_loss)
        results['validation_loss'].append(val_loss)
        results['tot_loss'].append(running_tot_loss)
        results['reg'].append(running_reg)
        results['l1'].append(running_l1)
        results['entropy'].append(running_entropy)
        
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
        
        test_loss = eval_model(model, test_data, t_test, criterion)
        log_message = f"Testing model on test dataset \nTest Loss: {test_loss}"
        save_logs(logs_file_path, log_message, save_updates)
        
    return results
        
        
        
        