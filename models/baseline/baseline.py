from models.utils.ODEBlock import ODEBlock
from torch_geometric.nn import TAGConv
import torch
import torch.nn.functional as F



class LB_ODE(ODEBlock):
    def __init__(
        self,
        model_path='./models'
        ):
        
        super().__init__(None, model_path, adjoint=False, integration_method='')
    
    
    def forward(self, snapshot):
        return snapshot.x
      
    def wrap_conv(self, conv):
        return conv
    
    def reset_params(self):
        return
    
    def regularization_loss(self, reg_loss_metrics):
        return 0.0
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        return
    
    
class TG_ODE(ODEBlock):
    def __init__(
        self, 
        model_path='./models',
        in_dim = 1,
        emb_dim = 32,
        K = 2,
        af = F.relu,
        step_size = 0.001,
        normalize = True,
        bias = True
    ):
        self.K = K
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.step_size = step_size
        self.normalize = normalize
        self.bias = bias
        
        conv = self._get_conv()
        super().__init__(conv, model_path, adjoint=False, integration_method='')
        
        self.emb_h = torch.nn.Linear(self.in_dim, self.emb_dim)
        self.readout = torch.nn.Linear(self.emb_dim, self.in_dim)
        self.af = af
        
    
    def wrap_conv(self, conv):
        return conv
    
    
    def forward(self, snapshot):
        edge_index, x, t = snapshot.edge_index, snapshot.x, snapshot.t_span
        delta_t = len(range(t.size(0))) - 1
        
        h = self.emb_h(x)
        
        for _ in range(delta_t):
            conv = self.conv(h, edge_index)
            h = h + self.step_size * self.af(conv)
        
        return self.readout(h)
    
    
    def _get_conv(self):
        return TAGConv(
            in_channels = self.emb_dim,
            out_channels = self.emb_dim,
            K = self.K,
            normalize = self.normalize,
            bias = self.bias
        )
        
    
    def reset_params(self):
        self.emb_h.reset_parameters()
        self.conv = self._get_conv()
        self.readout.reset_parameters() 
        
    
    def regularization_loss(self, reg_loss_metrics):
        return 0.0
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        return