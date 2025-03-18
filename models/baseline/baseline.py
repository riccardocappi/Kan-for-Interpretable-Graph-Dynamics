import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import MessagePassing
from models.utils.ModelInterface import ModelInterface
from utils.utils import save_black_box_to_file


class MLP(torch.nn.Module):
    """
    MLP Implementation
    """
    def __init__(self, hidden_layers, af, model_path, dropout_rate=0.0, save_black_box=False):
        super(MLP, self).__init__()
        self.af = af    # Activation function
        self.layers = torch.nn.ModuleList()
        self.model_path = model_path
        self.dropouts = torch.nn.ModuleList()
        
        for in_dim, out_dim in zip(hidden_layers, hidden_layers[1:]):
            self.layers.append(torch.nn.Linear(in_dim, out_dim))
            self.dropouts.append(torch.nn.Dropout(p=dropout_rate))
        
        self.cache_input = None
        self.cache_output = None
        self.save_black_box = save_black_box
    
    
    def forward(self, x: torch.Tensor):
        """
        MLP forward pass
        """
        if self.save_black_box:
            self.cache_input = x.detach()
        
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout except on the last layer
                x = self.af(x)
                x = dropout(x)
        
        if self.save_black_box:
            self.cache_output = x.detach()
        
        return x
            
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        self.mlp.save_black_box = True
        
        with torch.no_grad():
            _ = self.forward(dummy_x, dummy_edge_index)
        
        folder_path = f'{self.mlp.model_path}/cached_data'
        save_black_box_to_file(folder_path, self.mlp.cache_input, self.mlp.cache_output)
        
        

class MPNN(MessagePassing, ModelInterface):
    """
    Implementation of Message Passing Neural Network (MPNN) model
    """
    def __init__(self,
                 g_net:MLP,
                 h_net:MLP,
                 aggr = "add",
                 model_path='./models',
                 message_passing = True
                 ):
        
        MessagePassing.__init__(self, aggr=aggr)
        ModelInterface.__init__(self, model_path=model_path)
        
        self.h_net = h_net
        self.g_net = g_net
        self.message_passing = message_passing
                
    
    def forward(self, x, edge_index):
        """
        MPNN forward pass
        """
        return self.propagate(edge_index, x=x)
    
    
    def message(self, x_i, x_j):
        """
        Message function (implemented by G_Net)
        
        Args:
            -x_i : i-th node
            -x_j : j-th neighbor of i-th node
        """
        return self.g_net(torch.cat([x_j, x_i], dim=-1))
    
    
    def update(self, aggr_out, x):
        """
        Update function (implemented by H_Net)
        
        Args:
            -aggr_out : aggregation term
            -x : Input feature matrix
        """
        if self.message_passing:
            assert self.h_net.layers[0].in_features == 2 * x.size(1)
            return self.h_net(torch.cat([x, aggr_out], dim=-1))
        else:
            assert self.h_net.layers[0].in_features == x.size(1)
            return self.h_net(x) + aggr_out
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        """
        Implementation of ModelInterface method
        """
        return 0.0
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        """
        Implementation of ModelInterface method
        """
        self.g_net.save_black_box = True
        self.h_net.save_black_box = True
        
        with torch.no_grad():
            _ = self.forward(dummy_x, dummy_edge_index)
            
        save_black_box_to_file(
            folder_path=f'{self.g_net.model_path}/cached_data',
            cache_input=self.g_net.cache_input,
            cache_output=self.g_net.cache_output
        )
        
        save_black_box_to_file(
            folder_path=f'{self.h_net.model_path}/cached_data',
            cache_input=self.h_net.cache_input,
            cache_output=self.h_net.cache_output
        )
      
      
    def reset_params(self):
        """
        Implementation of ModelInterface method
        """
        for layer in self.g_net.layers:
            layer.reset_parameters()
            
        for layer in self.h_net.layers:
            layer.reset_parameters()