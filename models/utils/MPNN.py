from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.utils import degree
from typing import Union, Callable
from models.kan.KAN import KAN
from models.utils.MLP import MLP


class MPNN(MessagePassing):
    def __init__(
        self,
        g_net: Union[KAN, MLP, Callable],
        h_net: Union[KAN, MLP, Callable],
        aggr = "add",
        message_passing=True,
        include_time = False
        ):
        super().__init__(aggr=aggr)
        self.g_net = g_net
        self.h_net = h_net
        self.message_passing = message_passing
        self.include_time = include_time
        
    
    def forward(self, x, edge_index, edge_attr, t):
        
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        t_expanded = t.expand(x.size(0), 1) if self.include_time else torch.tensor([], device=t.device)
        
        if self.message_passing:
            out = self.h_net(torch.cat([x, t_expanded, self.g_net(aggr)], dim=-1))
        else:
            out = self.h_net(torch.cat([x, t_expanded], dim=-1)) + self.g_net(aggr)     
            
        return out
        
    
    def message(self, x_j, edge_attr):
        return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j