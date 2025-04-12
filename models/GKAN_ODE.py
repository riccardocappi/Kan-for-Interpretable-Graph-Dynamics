from .utils.ODEBlock import ODEBlock
from models.utils.MPNN import MPNN
import torch
import os
from utils.utils import save_acts, plot

class GKAN_ODE(ODEBlock):
    def __init__(
        self, 
        conv:MPNN, 
        model_path='./models', 
        adjoint=False, 
        integration_method='dopri5',
        lmbd_g=0.,
        lmbd_h=0.,
        **kwargs
    ):
        super().__init__(conv, model_path, adjoint, integration_method, **kwargs)
        self.lamb_g = lmbd_g
        self.lamb_h = lmbd_h
        
    
    def reset_params(self):
        for layer in self.conv.model.g_net.layers:
            layer.init_params()
        
        for layer in self.conv.model.h_net.layers:
            layer.init_params()
    
    
    def regularization_loss(self, reg_loss_metrics):
        reg_g, l1_g, entropy_g = self.conv.model.g_net.regularization_loss()
        reg_h, l1_h, entropy_h = self.conv.model.h_net.regularization_loss()
        
        # Update reg loss metrics 
        reg_loss_metrics['reg_g'] += reg_g.item()
        reg_loss_metrics['reg_h'] += reg_h.item()
        
        reg_loss_metrics['l1_g'] += l1_g.item()
        reg_loss_metrics['l1_h'] += l1_h.item()
        
        reg_loss_metrics['entropy_g'] += entropy_g.item()
        reg_loss_metrics['entropy_h'] += entropy_h.item()
        
        return (self.lamb_h * reg_h)+(self.lamb_g * reg_g)
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        self.eval()
        
        self.conv.model.g_net.store_act = True
        self.conv.model.h_net.store_act = True
        
        t = torch.tensor([], device=dummy_x.device) # Fake t for now
        
        with torch.no_grad():
            _ = self.conv.model.forward(dummy_x, dummy_edge_index, t=t)
        
        
        g_net_model_path = f"{self.model_path}/g_net"
        h_net_model_path = f"{self.model_path}/h_net"
        
        os.makedirs(g_net_model_path, exist_ok=True)
        os.makedirs(h_net_model_path, exist_ok=True)
        
        plot(folder_path=f'{h_net_model_path}/figures', layers=self.conv.model.h_net.layers, show_plots=False)
        plot(folder_path=f'{g_net_model_path}/figures', layers=self.conv.model.g_net.layers, show_plots=False)

        save_acts(layers=self.conv.model.h_net.layers, folder_path=f'{h_net_model_path}/cached_acts')
        save_acts(layers=self.conv.model.g_net.layers, folder_path=f'{g_net_model_path}/cached_acts') 