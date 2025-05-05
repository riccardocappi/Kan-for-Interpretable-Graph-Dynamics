import torch
from abc import ABC, abstractmethod
import os
from torchdiffeq import odeint, odeint_adjoint
from models.utils.MPNN import MPNN


class net_wrapper(torch.nn.Module):
    def __init__(self, model:MPNN):
        super().__init__()
        self.model = model
        self.edge_index = None
        self.edge_attr = None
        self.x_mean = None
        self.x_var = None

    
    def set_augmented_inputs(self, edge_index, edge_attr, x_mean, x_var):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x_mean = x_mean
        self.x_var = x_var
    
    
    def forward(self, t, x):
        return self.model(x, self.edge_index, self.edge_attr, t, x_var=self.x_var, x_mean=self.x_mean)
        
        
class ODEBlock(torch.nn.Module, ABC):
    def __init__(
        self,
        conv: torch.nn.Module,
        model_path = './models',
        adjoint = False,
        integration_method = 'dopri5',
        **kwargs
    ):
        super().__init__()
        self.conv = self.wrap_conv(conv)
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.adjoint = adjoint
        self.integration_method = integration_method
        self.odeint_function = odeint_adjoint if self.adjoint else odeint
        if self.adjoint:
            kwargs['adjoint_options'] = dict(norm="seminorm")
        if self.integration_method != 'dopri5':
            kwargs['options'] = dict(interp='linear')
            
        self.kwargs = kwargs
        
    
    def forward(self, snapshot):
        edge_index, edge_attr, x, t = snapshot.edge_index, snapshot.edge_attr, snapshot.x, snapshot.t_span
        
        # x shape (history, num_nodes, 1)
        
        x_mean, x_var = self.get_mean_var(x, snapshot.x_mask)
        
        self.conv.set_augmented_inputs(edge_index, edge_attr, x_mean, x_var)
        
        x = x[-1]   # Starting integration from the last timestamp of the input window.        
        integration = self.odeint_function(
            self.conv,
            x,
            t,
            method=self.integration_method,
            **self.kwargs
        )   # shape (horizon+1, num_nodes, 1)
        
        return integration[1:][snapshot.mask]
    
    
    def get_mean_var(self, x, x_mask):
        x_masked = x * x_mask  # zero out invalid values
        sum_valid = x_masked.sum(dim=0)  # (N, 1)
        count_valid = x_mask.sum(dim=0)  # (N, 1)
        mean = sum_valid / count_valid.clamp(min=1)  # (N, 1)
        
        # Compute masked variance
        squared_diff = ((x - mean)**2) * x_mask
        sum_squared_diff = squared_diff.sum(dim=0)
        variance = sum_squared_diff / count_valid.clamp(min=1)
        
        return mean, variance
    
    
    def wrap_conv(self, conv):
        return net_wrapper(conv)
    
        
    @abstractmethod
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        """
        Computes the regularization loss (e.g. L1 norm of model's weights. Can be also 0. for non-KAN-based models)
        Args:
            -reg_loss_metrics : dictionary in which to save metrics related to the regularization loss (e.g. the entropy term of the KAN reg loss)
        
        Returns: regularization loss
        """
        raise NotImplementedError()
    
    
    @abstractmethod
    def save_cached_data(self, dummy_x, dummy_edge_index):
        """
        This function is called in the post_processing step of Experiments, when saving model checkpoint. 
        Here you should save to file model's outputs and inputs that can be used later for symbolic regression.
        
        Args:
            dummy_x : Input for the forward pass of the model
            dummy_edge_index : Graph's edge_index for the forward pass of the model
        """
        raise NotImplementedError()
    
    
    @abstractmethod
    def reset_params(self):
        """
        reset the parameters of the model. This function is called to reset model's weights after each run in the 
        objective function of the Experiments class.
        """
        raise NotImplementedError()