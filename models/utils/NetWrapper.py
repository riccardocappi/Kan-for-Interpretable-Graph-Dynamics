import torch.nn as nn
from models.utils.ModelInterface import ModelInterface

class NetWrapper(nn.Module):
    """
    Just wraps around a torch module in order to properly work with the torchdiffeq methods.
    """
    def __init__(self, model:ModelInterface, edge_index, **kwargs):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.kwargs = kwargs
        
    
    def forward(self, t, x):
        """
        Forward pass
        
        Args:
            t : Current time step (mandatory for torchdiffeq odeint method)
            x : Input feature matrix. Shape = (num_nodes, in_dim)
        """
        return self.model(x, self.edge_index, **self.kwargs)
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        """
        Returns the model regularization loss
        """
        return self.model.regularization_loss(reg_loss_metrics)
               