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
        self.augmented_x = None

    
    def set_augmented_inputs(self, edge_index, edge_attr, augmented_x):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.augmented_x = augmented_x
    
    
    def forward(self, t, x):
        return self.model(x, self.edge_index, self.edge_attr, t, augmented_x=self.augmented_x)
        
        
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
        
        augmented_x = self.compute_node_features(x, snapshot.x_mask)  # shape (history, num_nodes, T*D + 2*D)
        
        self.conv.set_augmented_inputs(edge_index, edge_attr, augmented_x)
        
        x = x[-1]   # Starting integration from the last timestamp of the input window.        
        integration = self.odeint_function(
            self.conv,
            x,
            t,
            method=self.integration_method,
            **self.kwargs
        )   # shape (horizon+1, num_nodes, 1)
        
        return integration[1:][snapshot.mask]
    
    
    def teacher_forcing(self, x, y, t, teacher_forcing_ratio=1.0):
        predictions = []
        current_state = x
        for i in range(t.size(0)-1):
            t_span = t[i:i+2]   # (2,)
            pred = self.odeint_function(
                self.conv,
                current_state,
                t_span,
                method=self.integration_method,
                **self.kwargs
            )  # (2, N, 1)
            
            next_state = pred[-1]
            predictions.append(next_state)
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                current_state = y[i]  # Use ground truth
            else:
                current_state = next_state  # Use model prediction
        
    
    
    def compute_node_features(self, x: torch.Tensor, x_mask: torch.Tensor):
        dx_dt = x[1:] - x[:-1]                      # (T-1, N, D)
        dx_mask = x_mask[1:] * x_mask[:-1]  # mask only where both time steps are valid
        dx_dt = dx_dt * dx_mask
        
        # Reshape gradients: (N, (T-1)*D)
        dx_dt_flat = dx_dt.permute(1, 0, 2).reshape(x.shape[1], -1)
        
        # Compute mean and variance
        x_masked = x * x_mask
        sum_valid = x_masked.sum(dim=0)                      # (N, D)
        count_valid = x_mask.sum(dim=0).clamp(min=1)         # (N, D)
        mean = sum_valid / count_valid                       # (N, D)
        
        squared_diff = (torch.square(x - mean)) * x_mask
        variance = squared_diff.sum(dim=0) / count_valid     # (N, D)
        
        # Concatenate everything: (N, (T-1)*D + 2*D)
        features = torch.cat([dx_dt_flat, mean, variance], dim=1)
        return features
    
    
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