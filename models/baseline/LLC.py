from models.utils.ODEBlock import ODEBlock
from models.utils.MPNN import MPNN
import torch   
import os 
from utils.utils import save_black_box_to_file

class LLC_ODE(ODEBlock):
    def __init__(
        self, 
        conv:MPNN,
        model_path='./models', 
        adjoint=False, 
        integration_method='dopri5', 
        predict_deriv=False, 
        all_t=False,
        **kwargs
    ):
        self.last_y_pred, self.last_y_true = None, None # For penalization term computation
        super().__init__(conv, model_path, adjoint, integration_method, predict_deriv, all_t, **kwargs)
    
    
    def forward(self, snapshot):
        out = super().forward(snapshot)
        if self.predict_deriv and self.training:  # Penalization term is computed only when trained to predict dx/dt, as in the original paper
            self.last_y_true = snapshot.y
            self.last_y_pred = out
        return out
    
    
    def reset_params(self):
        llc_conv = self.conv.model # MPNN
        llc_conv.h_net.reset_params()
        llc_conv.g_net.reset_params()
        

    def regularization_loss(self, reg_loss_metrics):
        node_losses = torch.mean(torch.abs(self.last_y_true - self.last_y_pred), dim=1)
        variance = torch.var(node_losses)
        penalty = variance
        reg_loss_metrics["penalty_term"] = penalty.item()
        return penalty
            
    
    def save_cached_data(self, dummy_x, dummy_edge_index, dummy_t, dummy_edge_attr):
        self.eval()
        
        
        self.conv.model.g_net.save_black_box = True
        self.conv.model.h_net.f.save_black_box = True
        
        with torch.no_grad():
            # Forward pass
            _ = self.conv.model.forward(dummy_x, dummy_edge_index, edge_attr=dummy_edge_attr, t=dummy_t)
            
        g_net_model_path = f"{self.model_path}/g_net"
        h_net_model_path = f"{self.model_path}/h_net"
        
        os.makedirs(g_net_model_path, exist_ok=True)
        os.makedirs(h_net_model_path, exist_ok=True)
        
        assert (self.conv.model.g_net.cache_input is not None) and (self.conv.model.g_net.cache_output is not None)
        assert (self.conv.model.h_net.cache_input is not None) and (self.conv.model.h_net.cache_output is not None)
        
        save_black_box_to_file(
            folder_path=f'{g_net_model_path}/cached_data',
            cache_input=self.conv.model.g_net.cache_input,
            cache_output=self.conv.model.g_net.cache_output
        )
        
        save_black_box_to_file(
            folder_path=f'{h_net_model_path}/cached_data',
            cache_input=self.conv.model.h_net.cache_input,
            cache_output=self.conv.model.h_net.cache_output
        )
        
        
        
        
    
    
    