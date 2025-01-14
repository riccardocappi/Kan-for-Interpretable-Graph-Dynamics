from torch import nn
import torch
import os

from torch import nn
from .KanLayer import KANLayer
from models.KanConv import KANConv


class GKanConvLayer(nn.Module):
    
    def __init__(self, 
                 in_features,
                 out_features,
                 grid_size=5,
                 spline_order=3,
                 grid_range=[-1,1],
                 aggregate_first=True,
                 aggr='add',
                 norm=True,
                 scale_and_bias=False,
                 gconv_norm = 'A_hat_norm'
                 ):
        super(GKanConvLayer, self).__init__()
        
        self.kan_layer = KANLayer(
            in_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
            scale_and_bias=scale_and_bias
        )
        
        self.kan_conv = KANConv(
            in_features,
            out_features,
            k=spline_order,
            g=grid_size,
            grid_range=grid_range,
            aggregate_first=aggregate_first,
            aggr=aggr,
            norm=norm,
            scale_and_bias=scale_and_bias,
            norm_type=gconv_norm
        )
        
        
    def forward(self, x, edge_index, store_act=False):
        h = self.kan_layer(x, store_act) + self.kan_conv(x, edge_index, store_act)
        return h

    
    def regularization_loss(self, mu_1, mu_2, use_orig):
        if not use_orig:
            reg_l, l1_l, entropy_l = self.kan_layer.regularization_loss_fake(mu_1, mu_2)
            reg_c, l1_c, entropy_c = self.kan_conv.regularization_loss_fake(mu_1, mu_2)
        else:
            reg_l, l1_l, entropy_l = self.kan_layer.regularization_loss_orig(mu_1, mu_2)
            reg_c, l1_c, entropy_c = self.kan_conv.regularization_loss_orig(mu_1, mu_2)
        
        return reg_l+reg_c, l1_l+l1_c, entropy_l+entropy_c
    

class GKanConv_ODE(nn.Module):
    def __init__(self, hidden_layers, grid_size=5, spline_order=3, grid_range=[-1, 1], aggregate_first=True, 
                 model_path='./model', aggr='add', norm=True, epsilon = 1e-3, store_act=False, scale_and_bias=False,
                 gconv_norm = 'A_hat_norm', device='cuda'):
        
        super(GKanConv_ODE, self).__init__()
        
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(hidden_layers, hidden_layers[1:]):
            self.layers.append(
                GKanConvLayer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    aggregate_first=aggregate_first,
                    aggr=aggr,
                    norm=norm,
                    scale_and_bias=scale_and_bias,
                    gconv_norm=gconv_norm
                )
            )
        
        
        self.epsilon = epsilon
        self.device = device
        self.model_path = model_path
        self.store_act = store_act
        self.cache_data = None
        
        self.to(self.device)
        
        
    
    def to(self, device):
        super(GKanConv_ODE, self).to(device)
        self.device = device
        for layer in self.layers:
            layer.to(device)
    
    
    def _forward(self, h, edge_index):
        # GKanConv forward pass
        for layer in self.layers:
            h = layer(h, edge_index, self.store_act)   
        return h
    
    
    def forward(self, data):
        # GKanConv-ODE forward pass
        x, edge_index, delta_t = data.x, data.edge_index, data.delta_t
        
        if self.cache_data is None:
            self.cache_data = data.detach()
        
        for _ in range(delta_t):
            x = x + self.epsilon * self._forward(x, edge_index)
        
        return x


    def regularization_loss(self, mu_1=1.0, mu_2=1.0, use_orig=False):
        tot_reg, tot_l1, tot_entropy = 0., 0., 0.
        for layer in self.layers:
            reg, l1, entropy = layer.regularization_loss(mu_1, mu_2, use_orig)
            tot_reg += reg
            tot_l1 += l1
            tot_entropy += entropy
        return tot_reg, tot_l1, tot_entropy