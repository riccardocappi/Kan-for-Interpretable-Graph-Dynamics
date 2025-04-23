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
        
    
    def forward(self, t, x):
        return self.model(x, self.edge_index, self.edge_attr, t)
        
        
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
        
        if hasattr(self.conv, 'edge_index'):
            if self.conv.edge_index is None:
                self.conv.edge_index = edge_index
        
        if hasattr(self.conv, 'edge_attr'):
            if self.conv.edge_attr is None:
                self.conv.edge_attr = edge_attr
         
        integration = self.odeint_function(
            self.conv,
            x,
            t,
            method=self.integration_method,
            **self.kwargs
        )
        
        return integration[1:]
    
    
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