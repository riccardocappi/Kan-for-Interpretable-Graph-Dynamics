import torch.nn as nn


class NetWrapper(nn.Module):
    def __init__(self, model, model_config, edge_index, **kwargs):
        super().__init__()
        self.model = model(**model_config)
        self.edge_index = edge_index
        self.kwargs = kwargs
        
    
    def forward(self, t, x):
        return self.model(x, self.edge_index, **self.kwargs)
    
    
    def regularization_loss(self, mu_1, mu_2, use_orig=False):
        return self.model.regularization_loss(mu_1, mu_2, use_orig)
    
    
    def to(self, device):
        super().to(device)
        self.model.to(device)
               