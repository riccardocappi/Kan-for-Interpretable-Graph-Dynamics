import torch
from utils.utils import save_black_box_to_file
from models.utils.ODEBlock import ODEBlock
import os
from models.utils.MPNN import MPNN


class MPNN_ODE(ODEBlock):
    def __init__(
        self, 
        conv: MPNN, 
        model_path='./models', 
        adjoint=False, 
        integration_method='dopri5',
        **kwargs
    ):
        super().__init__(conv, model_path, adjoint, integration_method, **kwargs)
        
    
    def reset_params(self):
        for layer in self.conv.model.g_net.layers:
            layer.reset_parameters()
            
        for layer in self.conv.model.h_net.layers:
            layer.reset_parameters()
    
    
    def regularization_loss(self, reg_loss_metrics):
        return 0.0
            
    
    def save_cached_data(self, dummy_x, dummy_edge_index, dummy_t, dummy_edge_attr):
        self.eval()
        
        self.conv.model.g_net.save_black_box = True
        self.conv.model.h_net.save_black_box = True
                
        with torch.no_grad():
            _ = self.conv.model.forward(dummy_x, dummy_edge_index, edge_attr=dummy_edge_attr, t=dummy_t)
        
        
        g_net_model_path = f"{self.model_path}/g_net"
        h_net_model_path = f"{self.model_path}/h_net"
        
        os.makedirs(g_net_model_path, exist_ok=True)
        os.makedirs(h_net_model_path, exist_ok=True)
        
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