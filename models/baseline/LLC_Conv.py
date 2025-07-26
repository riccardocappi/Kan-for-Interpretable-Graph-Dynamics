import torch
from models.utils.MLP import MLP
from typing import List


class Q_inter(torch.nn.Module):
    def __init__(
        self,
        hidden_layers_g0:List,
        hidden_layers_g1: List,
        hidden_layers_g2: List,
        af_g0,
        af_g1,
        af_g2,
        dr_g0 = 0.0,
        dr_g1 = 0.0,
        dr_g2 = 0.0,
        save_black_box=False
    ):
        super().__init__()
        
        self.g0 = MLP(
            hidden_layers=hidden_layers_g0,
            af = af_g0,
            dropout_rate=dr_g0
        )
                
        self.g1 = MLP(
            hidden_layers=hidden_layers_g1,
            af = af_g1,
            dropout_rate=dr_g1
        )
        
        self.g2 = MLP(
            hidden_layers=hidden_layers_g2,
            af = af_g2,
            dropout_rate=dr_g2
        )
        
        self.save_black_box = save_black_box
        self.cache_input, self.cache_output = None, None
    
    
    def forward(self, x: torch.Tensor):
        # x has shape (batch, in_dim*2)
        if self.save_black_box:
            self.cache_input = x.detach()
            
        in_dim = x.shape[1] // 2
        x_i = x[:, :in_dim]   # First half
        x_j = x[:, in_dim:]   # Second half
        
        out_g0 = self.g0(x)
        out_g1 = self.g1(x_i)
        out_g2 = self.g2(x_j)
        
        out = out_g0 + (out_g1 * out_g2)
        
        if self.save_black_box:
            self.cache_output = out.detach()
        
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
        dropout_rate = 0.0,
        save_black_box = False
    ):
        super().__init__()
        self.f = MLP(
            hidden_layers=hidden_layers,
            af=af,
            dropout_rate=dropout_rate,
            save_black_box=save_black_box
        )
        
        self.cache_input = None
        self.cache_output = None
    
    def forward(self, x: torch.Tensor):
        out = self.f(x)
        self.cache_input = self.f.cache_input
        self.cache_output = self.f.cache_output
        return out
    
    def reset_params(self):
        for layer in self.f.layers:
            layer.reset_parameters()
    