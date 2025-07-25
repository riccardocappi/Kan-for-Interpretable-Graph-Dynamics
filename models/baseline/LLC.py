import torch
from models.utils.MLP import MLP
from typing import List
from models.utils.ODEBlock import ODEBlock
from models.utils.MPNN import MPNN

class Q_inter(torch.nn.Module):
    def __init__(
        self,
        hidden_layers:List,
        af,
        dropout_rate = 0.0
    ):
        super().__init__()
        
        in_dim = hidden_layers[0]
        self.g0 = MLP(
            hidden_layers=hidden_layers,
            af = af,
            dropout_rate=dropout_rate
        )
        
        assert in_dim % 2 == 0, "Pairwise interaction network must have in dimension 2 * x_i dim"
        
        hidden_layers_g1 = hidden_layers
        hidden_layers_g1[0] = in_dim / 2
        
        self.g1 = MLP(
            hidden_layers=hidden_layers_g1,
            af = af,
            dropout_rate=dropout_rate
        )
        
        self.g2 = MLP(
            hidden_layers=hidden_layers_g1,
            af = af,
            dropout_rate=dropout_rate
        )
    
    
    def forward(self, x):
        # x has shape (batch, in_dim*2)
        in_dim = x.shape[1] // 2
        x_i = x[:, :in_dim]   # First half
        x_j = x[:, in_dim:]   # Second half
        
        out_g0 = self.g0(x)
        out_g1 = self.g1(x_i)
        out_g2 = self.g2(x_j)
        
        out = out_g0 + (out_g1 * out_g2)
        return out
    
    
    def reset_params(self):
        for g in [self.g0, self.g1, self.g2]:
            for layer in g.layers:
                layer.reset_parameters()
                


class Q_self(torch.nn.Module):
    def __init__(
        self,
        hidden_layers:List,
        af,
        dropout_rate = 0.0
    ):
        super().__init__()
        self.f = MLP(
            hidden_layers=hidden_layers,
            af=af,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x):
        return self.f(x)
    
    def reset_params(self):
        for layer in self.f.layers:
            layer.reset_parameters()
    
    

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
        super().__init__(conv, model_path, adjoint, integration_method, predict_deriv, all_t, **kwargs)
    
    
    def forward(self, snapshot):
        return super().forward(snapshot)
    
    
    def reset_params(self):
        llc_conv = self.conv.model # MPNN
        llc_conv.h_net.reset_params()
        llc_conv.g_net.reset_params()
        

    def regularization_loss(self, reg_loss_metrics):
        pass
            
    
    def save_cached_data(self, dummy_x, dummy_edge_index, dummy_t, dummy_edge_attr):
        pass
    
    
    