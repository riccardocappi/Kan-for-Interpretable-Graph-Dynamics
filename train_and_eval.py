import torch
from utils.utils import sample_temporal_graph, get_temporal_data_loader, save_logs, get_acts, plot, save_acts
from models.GKanConv_ODE import GKanConv_ODE
import os
from torch.optim import LBFGS
import json




def eval_model(model, loader, criterion):
    model.eval()
    running_val_loss = 0.
    count = 0
    with torch.no_grad():
        for batch in loader:
            y_true, y_pred = [], []
            for snapshot in batch:
                out_val = model.forward(snapshot)
                y_pred.append(out_val)
                y_true.append(snapshot.y)
            
            y_pred = torch.cat(y_pred)
            y_true = torch.cat(y_true) 

            val_loss = criterion(y_pred, y_true)
            running_val_loss += val_loss.item()
            count += 1
    return running_val_loss/count


def fit(model, train_dataset, val_dataset, test_dataset, seed=42, epochs=50, patience=30, lr = 0.001, lamb=0., batch_size=32,
        log=10, mu_1=1.0, mu_2=1.0, log_file_name='logs.txt', criterion = torch.nn.L1Loss(), opt='Adam', 
        use_orig_reg=False, save_updates=True, update_grid=False):


    torch.manual_seed(seed)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    batch_size_train = train_size if batch_size == -1 else batch_size
    batch_size_val = val_size if batch_size == -1 else batch_size
            
    train_loader = get_temporal_data_loader(train_dataset, model.device, batch_size_train)
    val_loader = get_temporal_data_loader(val_dataset, model.device, batch_size_val)
    test_loader = get_temporal_data_loader(test_dataset, model.device, batch_size_val)
    
    # if update_grid:
    #    update_grid_from_samples(model, train_dataset, 32)
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'LBFGS':
        optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    else:
        raise Exception('Optimizer not implemented yet!')
        
    logs_folder = f'{model.model_path}/logs'
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
        
    global running_training_loss, running_tot_loss, running_reg, running_l1, running_entropy
    
    def training():
        global running_training_loss, running_tot_loss, running_reg, running_l1, running_entropy
        optimizer.zero_grad()
        y_pred, y_true = [], []
        for snapshot in batch:
            out_tr = model.forward(snapshot)
            y_pred.append(out_tr)
            y_true.append(snapshot.y)
        
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
            
        training_loss = criterion(y_pred, y_true)
        running_training_loss = running_training_loss + training_loss.item()
        reg, l1, entropy = model.regularization_loss(mu_1, mu_2, use_orig_reg)
        loss = training_loss + lamb * reg
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
        count = 0
        for batch in train_loader:
            if opt == 'Adam':
                _ = training()
            else:
                optimizer.step(training)
            count += 1
        train_loss = running_training_loss / count
        tot_loss = running_tot_loss / count
        
        val_loss = eval_model(model, val_loader, criterion)
        results['train_loss'].append(train_loss)
        results['validation_loss'].append(val_loss)
        results['tot_loss'].append(tot_loss)
        results['reg'].append(running_reg / count)
        results['l1'].append(running_l1 / count)
        results['entropy'].append(running_entropy/count)
        
        if epoch % log == 0:
            log_message = f"Epoch: {epoch} \t Training loss: {train_loss:0.5f} \t Val Loss: {val_loss:0.5f} \t Tot Loss: {tot_loss:0.5f}"
            # print(log_message)
            save_logs(logs_file_path, log_message, save_updates)
        if val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        elif epoch - best_epoch > patience:
            log_message = f"Early stopping at epoch {epoch}"
            # print(log_message)
            save_logs(logs_file_path, log_message, save_updates)
            break
        
        
    log_message = f"\nLoading best model found at epoch {best_epoch} with val loss {best_val_loss}" 
    # print(log_message)
    save_logs(logs_file_path, log_message, save_updates)
    model.load_state_dict(best_model_state)
    if save_updates:
        torch.save(best_model_state, f'{model.model_path}/best_state_dict.pth')
        with open(f"{model.model_path}/results.json", "w") as outfile: 
            json.dump(results, outfile)
        
        test_loss = eval_model(model, test_loader, criterion)
        log_message = f"Testing model on test dataset \nTest Loss: {test_loss}"
        # print(log_message)
        save_logs(logs_file_path, log_message)
    return results



def train_and_eval(hidden_layers, model_path, aggregate_first, input_range, aggr, norm, epsilon, device, gconv_norm, 
                   grid_size, spline_order, train_dataset, valid_dataset, test_dataset, batch, epochs, lr, 
                   opt, log, patience, lamb, mu_1, mu_2, eval_model, criterion=torch.nn.L1Loss(), 
                   use_orig_reg=False, sample_size=32, show_plots=False, store_acts=False, update_grid=False):
        
    model = GKanConv_ODE(
        hidden_layers,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=[input_range[0], input_range[1]],
        aggregate_first=aggregate_first,
        model_path=model_path,
        aggr=aggr,
        norm=norm,
        epsilon=epsilon,
        store_act=store_acts,
        gconv_norm=gconv_norm,
        device=device
    )
    
    results = fit(model, train_dataset, valid_dataset, test_dataset, batch_size=batch, epochs=epochs, 
                  criterion=criterion, lr=lr, opt = opt, log=log, patience=patience, lamb=lamb, 
                  use_orig_reg=use_orig_reg, save_updates=eval_model, mu_1=mu_1, mu_2=mu_2, update_grid=update_grid)
    
    best_val_loss = min(results['validation_loss'])
       
    if eval_model:
        # Obtains a samples from the training set by sampling graph snapshots at different time-steps
        model.store_act = True
        dummy_x = sample_temporal_graph(train_dataset, device=device, sample_size=sample_size)
        get_acts(model, dummy_x)
        
        folder_kan_layers = f"{model.model_path}/kan_layers"
        folder_kan_convs = f"{model.model_path}/kan_convs"
        
        plot(f'{folder_kan_layers}/figures', [l.kan_layer for l in model.layers], show_plots)       
        plot(f'{folder_kan_convs}/figures', [l.kan_conv for l in model.layers], show_plots)
        
        save_acts([l.kan_layer for l in model.layers], f'{folder_kan_layers}/cached_acts')
        save_acts([l.kan_conv for l in model.layers], f'{folder_kan_convs}/cached_acts')
        
        torch.save(dummy_x, f"{model.model_path}/sample")
         
    
    return best_val_loss

