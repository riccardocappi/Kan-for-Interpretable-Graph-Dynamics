from .kan.KAN import KAN
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.utils import degree
from models.utils.ModelInterface import ModelInterface
from utils.utils import plot, save_acts
import os


class GKAN_ODE(MessagePassing, ModelInterface):
    def __init__(self,
                 h_net: KAN,
                 g_net: KAN,
                 model_path = './model',
                 norm=False,
                 device='cuda',
                 lmbd_g=0.,
                 lmbd_h=0.,
                 ):
        
        MessagePassing.__init__(self, aggr='add')
        ModelInterface.__init__(self, model_path=model_path)
        
        
        self.h_net = h_net
        
        self.g_net = g_net
        
        self.device = torch.device(device)
        self.norm = norm
        self.lmbd_g = lmbd_g
        self.lmbd_h = lmbd_h
               
        self.to(self.device)


    def to(self, device):
        super().to(device)
        self.device = device
        self.h_net.to(device)
        self.g_net.to(device)
        
    
    def forward(self, x, edge_index, update_grid=False):
        norm = self.get_norm(edge_index, x) if self.norm else torch.ones(edge_index.shape[1], device=x.device)
        
        return self.propagate(edge_index, x=x, norm=norm, update_grid=update_grid)
    

    def message(self, x_i, x_j, norm, update_grid):
        mes = self.g_net(torch.cat([x_j, x_i], dim=-1), update_grid=update_grid)
        return norm.view(-1, 1) * mes
        
    
    def update(self, aggr_out, x, update_grid):
        return self.h_net(torch.cat([x, aggr_out], dim=-1), update_grid=update_grid)
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        reg_g, l1_g, entropy_g = self.g_net.regularization_loss()
        reg_h, l1_h, entropy_h = self.h_net.regularization_loss()
        
        # Update reg loss metrics 
        reg_loss_metrics['reg_g'] += reg_g.item()
        reg_loss_metrics['reg_h'] += reg_h.item()
        
        reg_loss_metrics['l1_g'] += l1_g.item()
        reg_loss_metrics['l1_h'] += l1_h.item()
        
        reg_loss_metrics['entropy_g'] += entropy_g.item()
        reg_loss_metrics['entropy_h'] += entropy_h.item()
        
        return (self.lmbd_h * reg_h)+(self.lmbd_g * reg_g)
        # return reg_h + reg_g
       
    
    def get_norm(self, edge_index, x):
        row, _ = edge_index  # Use the source nodes
        deg = degree(row, x.size(0), dtype=x.dtype)  # Compute degree for source nodes
        norm = 1. / deg[row]  # Normalize using the source node degree
        return norm
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        self.g_net.store_act = True
        self.h_net.store_act = True
        
        with torch.no_grad():
            _ = self.forward(dummy_x, dummy_edge_index, update_grid=False)
        
        
        g_net_model_path = f"{self.model_path}/g_net"
        h_net_model_path = f"{self.model_path}/h_net"
        
        os.makedirs(g_net_model_path, exist_ok=True)
        os.makedirs(h_net_model_path, exist_ok=True)
        
        plot(folder_path=f'{h_net_model_path}/figures', layers=self.h_net.layers, show_plots=False)
        plot(folder_path=f'{g_net_model_path}/figures', layers=self.g_net.layers, show_plots=False)

        save_acts(layers=self.h_net.layers, folder_path=f'{h_net_model_path}/cached_acts')
        save_acts(layers=self.g_net.layers, folder_path=f'{g_net_model_path}/cached_acts') 
        
        
    def reset_params(self):
        for layer in self.g_net.layers:
            layer.init_params()
        
        for layer in self.h_net.layers:
            layer.init_params()
    
            
    
        


