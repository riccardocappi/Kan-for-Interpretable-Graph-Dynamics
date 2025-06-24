import os
import torch
from .KanLayer import KANLayer
import dill


class KAN(torch.nn.Module):
    '''
    Implementation from scratch of KAN model. 
    '''
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        store_act = False,
        device='cuda',
        mu_1 = 1.,
        mu_2 = 1.,
        use_orig_reg = False,
        compute_symbolic = False,
        compute_mult = False
        ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.cache_data = None
        self.store_act = store_act
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.hidden_layer = layers_hidden
        self.spline_order = spline_order
        self.device = torch.device(device)
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.use_orig_reg = use_orig_reg
        
        
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLayer(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    compute_symbolic=compute_symbolic,
                    compute_mult=compute_mult
                    )
            )
        self.to(self.device)
        
        
      
    def to(self, device):
        '''
        Send model to the specified device
        '''
        super(KAN, self).to(device)
        self.device = device
        for layer in self.layers:
            layer.to(device)



    def forward(self, x: torch.Tensor, update_grid=False):
        '''
        KAN forward pass
        '''
        if self.cache_data is None:
            self.cache_data = x.detach()
        
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
                
            x = layer(x, self.store_act)
        return x


    def regularization_loss(self):
        '''
        Returns the regularization loss. You can either decide to use the original definition or the one
        proposed by the authors of efficient-kan
        '''
        tot_reg, tot_l1, tot_entropy = 0., 0., 0.
        for layer in self.layers:
            if not self.use_orig_reg:
                reg, l1, entropy = layer.regularization_loss_fake(self.mu_1, self.mu_2)
            else:
                reg, l1, entropy = layer.regularization_loss_orig(self.mu_1, self.mu_2)
            tot_reg += reg
            tot_l1 += l1
            tot_entropy += entropy
        return tot_reg, tot_l1, tot_entropy
                    
                    
                    
    
        
    
