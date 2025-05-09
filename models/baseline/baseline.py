from models.utils.ODEBlock import ODEBlock
# from torch_geometric.nn import TAGConv
# import torch
# import torch.nn.functional as F



class LB_ODE(ODEBlock):
    def __init__(
        self,
        model_path='./models'
        ):
        
        super().__init__(None, model_path, adjoint=False, integration_method='')
    
    
    def forward(self, snapshot):
        x = snapshot.x[-1]
        return x.unsqueeze(0).repeat(snapshot.y.size(0), 1, 1)
    
    
    def wrap_conv(self, conv):
        return conv
    
    def reset_params(self):
        return
    
    def regularization_loss(self, reg_loss_metrics):
        return 0.0
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        return