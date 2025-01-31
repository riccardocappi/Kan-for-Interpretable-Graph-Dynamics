from .Experiments import Experiments
from models.utils.NetWrapper import NetWrapper
from models.baseline.baseline import GCN, GIN
import torch
from train_and_eval import fit


class ExperimentsBaseline(Experiments):
    
    def __init__(self, config, G, n_trials, model_selection_method='optuna', model_type='GCN'):
        super().__init__(config, G, n_trials, model_selection_method)
        self.model_type = model_type
    
    
    def pre_processing(self, train_data, valid_data):
        return train_data, valid_data
    
    
    def objective(self, trial):
        if self.model_type == 'GCN':
            model = self._get_GCN(trial)
        elif self.model_type == 'GIN':
            model = self._get_GIN(trial)
        else:
            raise Exception('Model not supported')

        lr = trial.suggest_float('lr', self.search_space['lr'][0], self.search_space['lr'][-1])
        
        results = fit(
            model,
            self.training_set,
            self.valid_set,
            epochs=self.epochs,
            patience=self.patience,
            lr = lr,
            log=self.log,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            save_updates=False,
            n_iter=self.n_iter,
            batch_size=-1,
            t_f_train=self.t_f_train
        )
        
        best_val_loss = min(results['validation_loss']) 
        return best_val_loss
        

    def _get_GCN(self, trial):
        hidden_dims = trial.suggest_int('hidden_dim', self.search_space['hidden_dim'][0], self.search_space['hidden_dim'][-1])
        model_config = {
            'input_dim': self.config['in_dim'],
            'hidden_dim': hidden_dims,
            'output_dim': self.config['in_dim'],
            'model_path': self.model_path
        }
        
        model = NetWrapper(GCN, model_config, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model
    
    
    def _get_GIN(self, trial):
        hidden_dims = trial.suggest_int('hidden_dim', self.search_space['hidden_dim'][0], self.search_space['hidden_dim'][-1])
        epsilon = trial.suggest_float('epsilon', self.search_space['epsilon'][0], self.search_space['epsilon'][-1])
        
        model_config = {
            'input_dim': self.config['in_dim'],
            'hidden_dim': hidden_dims,
            'output_dim': self.config['in_dim'],
            'model_path': self.model_path,
            'epsilon': epsilon
        }
        model = NetWrapper(GIN, model_config, self.edge_index)
        model = model.to(torch.device(self.device))
        
        return model

    
    def post_processing(self, best_params):
        model_config = {}
        for key in self.search_space.keys():
            if key != 'lr':
                model_config[key] = best_params[key]
            
        model_config['model_path'] = f'{self.model_path}/eval'
        model_config['input_dim'] = self.config['in_dim']
        model_config['output_dim'] = self.config['in_dim']
        
        net = GCN if self.model_type == 'GCN' else GIN          #TODO: Generalize this
        model = NetWrapper(net, model_config, self.edge_index)
        model = model.to(torch.device(self.device))
        
        _ = fit(
            model,
            self.training_set,
            self.valid_set,
            epochs=self.epochs,
            patience=self.patience,
            lr = best_params['lr'],
            lmbd=1.,
            log=self.log,
            criterion=torch.nn.MSELoss(),
            opt=self.opt,
            save_updates=True,
            n_iter=self.n_iter,
            batch_size=-1,
            t_f_train=self.t_f_train
        )
        
        return model