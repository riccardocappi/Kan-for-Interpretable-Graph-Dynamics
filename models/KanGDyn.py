from .KAN import KAN
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.utils import degree

class KanGDyn(MessagePassing):
    def __init__(self, 
                 h_hidden_layers, 
                 g_hidden_layers,
                 grid_size=5,
                 spline_order=3,
                 grid_range = [-1, 1],
                 model_path = './model',
                 store_acts = False,
                 scale_and_bias = False,
                 norm=False,
                 device='cuda'):
        
        super(KanGDyn, self).__init__(aggr='add')
        
        # self.f_net = KAN(f_hidden_layers,
        #                 grid_size=grid_size,
        #                 spline_order=spline_order,
        #                 grid_range=grid_range,
        #                 model_path=f'{model_path}/F_Net',
        #                 store_act=store_acts,
        #                 scale_and_bias=scale_and_bias,
        #                 device=device)
        
        
        self.h_net = KAN(h_hidden_layers,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         grid_range=grid_range,
                         model_path=f'{model_path}/H_Net',
                         store_act=store_acts,
                         scale_and_bias=scale_and_bias,
                         device=device)
        
        self.g_net = KAN(g_hidden_layers,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         grid_range=grid_range,
                         model_path=f'{model_path}/G_Net',
                         store_act=store_acts,
                         scale_and_bias=scale_and_bias,
                         device=device)
        
        self.model_path = model_path
        self.device = device
        self.norm = norm
        
        self.to(self.device)
        

    def to(self, device):
        super().to(device)
        self.device = device
        # self.f_net.to(device)
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
    
    
    def regularization_loss(self, mu_1, mu_2, use_orig=False):
        reg_g, l1_g, entropy_g = self.g_net.regularization_loss(mu_1, mu_2, use_orig)
        reg_h, l1_h, entropy_h = self.h_net.regularization_loss(mu_1, mu_2, use_orig)
        return reg_h+reg_g, l1_g+l1_h, entropy_h+entropy_g
    
    
    def get_norm(self, edge_index, x):
        row, _ = edge_index  # Use the source nodes
        deg = degree(row, x.size(0), dtype=x.dtype)  # Compute degree for source nodes
        norm = 1. / deg[row]  # Normalize using the source node degree
        return norm
        


