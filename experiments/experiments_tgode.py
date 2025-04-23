from .Experiments import Experiments
# from datasets.SpatioTemporalGraph import SpatioTemporalGraph
# from datasets.TrafficData import traffic_data_name
# import torch
# from tsl.data.preprocessing.scalers import MinMaxScaler
# from models.baseline.baseline import TG_ODE
# from .experiments_mpnn import activations


class ExperimentsTGODE(Experiments):
    pass

# class ExperimentsTGODE(Experiments):
#     def __init__(
#         self, 
#         config,
#         n_trials, 
#         model_selection_method='optuna',
#         study_name='example',
#         process_id=0,
#         store_to_sqlite=False
#     ):
#         super().__init__(config, n_trials, model_selection_method, study_name, process_id, store_to_sqlite)

    
#     def pre_processing(self, training_set:SpatioTemporalGraph):
#         scaler = None
#         if self.config['name'] in traffic_data_name:
#             all_train_x = torch.cat([data.x for data in training_set], dim=0)
            
#             scaler = MinMaxScaler(out_range=(-1, 1))
#             scaler.fit(all_train_x.detach().cpu())
            
#             scaler.scale = scaler.scale.to(torch.device(self.device))
#             scaler.bias = scaler.bias.to(torch.device(self.device))
            
#         return scaler

    
#     def get_model_opt(self, trial):
#         model_path = f'{self.model_path}/tgode'
#         in_dim = self.config['in_dim']
#         emb_dim = trial.suggest_categorical(
#             'emb_dim',
#             self.search_space['emb_dim']
#         )
#         K = trial.suggest_int(
#             'K',
#             self.search_space['K'][0],
#             self.search_space['K'][-1]
#         )
#         activation = trial.suggest_categorical(
#             'af',
#             self.search_space['af']
#         )
#         af = activations[activation]
        
#         step_size = trial.suggest_float(
#             'step_size',
#             self.search_space['step_size'][0],
#             self.search_space['step_size'][-1]
#         )
        
#         normalize = trial.suggest_categorical(
#             'normalize',
#             self.search_space['normalize']
#         )
        
#         bias = trial.suggest_categorical(
#             'bias',
#             self.search_space['bias']
#         )
        
#         model = TG_ODE(
#             model_path=model_path,
#             in_dim=in_dim,
#             emb_dim=emb_dim,
#             K=K,
#             af=af,
#             step_size=step_size,
#             normalize=normalize,
#             bias=bias
#         )
        
#         model = model.to(torch.device(self.device))
#         return model
        
        