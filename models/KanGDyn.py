import torch
from .KAN import KAN
from torch_geometric.nn import MessagePassing


class KanGDyn(MessagePassing):
    def __init__(self, f_hidden_layers, h_hidden_layers, g_hidden_layers):  # TODO: Add other arguments
        super(KanGDyn, self).__init__(aggr='add')
        
        self.f_net = KAN(f_hidden_layers)
        self.h_net = KAN(h_hidden_layers)
        self.g_net = KAN(g_hidden_layers)
        

    def forward(self, x, edge_index):
        self_interaction = self.f_net(x)
        
        out = self.propagate(edge_index, x=x)
        
        return self_interaction + out
    

    def message(self, x_i, x_j):
        return self.g_net(x_i, x_j)
        
    
    def update(self, aggr_out, x):
        return self.h_net(x, aggr_out)
    
    
    def aggregate(self, inputs, index):
        return inputs
        
        
        


