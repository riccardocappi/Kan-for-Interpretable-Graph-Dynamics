import torch.nn as nn

class NetWrapper(nn.Module):
    def __init__(self, model, edge_index, **kwargs):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.kwargs = kwargs
        
    
    def forward(self, t, x):
        return self.model(x, self.edge_index, **self.kwargs)
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        return self.model.regularization_loss(reg_loss_metrics)
               