import os
import yaml
import torch
from .KanLayer import KANLayer


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
        model_path = './model',
        store_act = False,
        scale_and_bias=False,
        device='cuda'
        ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.cache_data = None
        self.model_path = model_path
        self.store_act = store_act
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.hidden_layer = layers_hidden
        self.spline_order = spline_order
        self.device = torch.device(device)
        
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
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
                    scale_and_bias=scale_and_bias
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



    def forward(self, x: torch.Tensor):
        '''
        KAN forward pass
        '''
        if self.cache_data is None:
            self.cache_data = x.detach()
        
        for layer in self.layers:
            x = layer(x, self.store_act)
        return x


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, use_orig=False):
        '''
        Returns the regularization loss. You can either decide to use the original definition or the one
        proposed by the authors of efficient-kan
        '''
        tot_reg, tot_l1, tot_entropy = 0., 0., 0.
        for layer in self.layers:
            if not use_orig:
                reg, l1, entropy = layer.regularization_loss_fake(regularize_activation, regularize_entropy)
            else:
                reg, l1, entropy = layer.regularization_loss_orig(regularize_activation, regularize_entropy)
            tot_reg += reg
            tot_l1 += l1
            tot_entropy += entropy
        return tot_reg, tot_l1, tot_entropy
                         
                
    def to_original_kan(self):
        '''
        Save the models parameters so that they can be loaded to a corresponding model based on the original
        KAN implementation.
        '''
        folder = f'{self.model_path}/original-kan-state'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        state_dict = {}
        for l, layer in enumerate(self.layers):
            subnode_bias = torch.nn.Parameter(torch.zeros(layer.out_features)).to(self.device).detach().requires_grad_(False)
            subnode_scale = torch.nn.Parameter(torch.ones(layer.out_features)).to(self.device).detach().requires_grad_(False)
            
            state_dict[f'subnode_bias_{l}'] = subnode_bias
            state_dict[f'subnode_scale_{l}'] = subnode_scale
            
            # KAN layers params
            orig_state_dict = self.state_dict()
            
            state_dict[f'node_bias_{l}'] = orig_state_dict[f'layers.{l}.layer_bias']
            state_dict[f'node_scale_{l}'] = orig_state_dict[f'layers.{l}.layer_scale']
            
            state_dict[f'act_fun.{l}.grid'] = orig_state_dict[f'layers.{l}.grid']
            state_dict[f'act_fun.{l}.coef'] = orig_state_dict[f'layers.{l}.spline_weight'].permute(1, 0, 2)
            
            state_dict[f'act_fun.{l}.mask'] =  orig_state_dict[f'layers.{l}.layer_mask'].permute(1, 0)
            
            state_dict[f'act_fun.{l}.scale_base'] = orig_state_dict[f'layers.{l}.base_weight'].permute(1, 0)
            
            state_dict[f'act_fun.{l}.scale_sp'] = orig_state_dict[f'layers.{l}.spline_scaler'].permute(1, 0)
            
            # Symbolic layer params
            state_dict[f'symbolic_fun.{l}.mask'] = orig_state_dict[f'layers.{l}.symb_mask']
            state_dict[f'symbolic_fun.{l}.affine'] = orig_state_dict[f'layers.{l}.affine_params']
        
        torch.save(state_dict, f'{folder}/original-kan-state.pth')
        torch.save(self.cache_data, f'{folder}/cache_data')
        config = dict(
            hidden_layers = self.hidden_layer,
            grid_size = self.grid_size,
            grid_range = self.grid_range,
            spline_order = self.spline_order
        )
        for l, layer in enumerate(self.layers):
            config[f'symbolic_functions_layer_{l}'] = layer.symb_dict_names
        with open(f'{folder}/config.yml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
    
