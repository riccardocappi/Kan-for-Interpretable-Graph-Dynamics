from .KanLayer import KANLayer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch


class GraphConv(MessagePassing):
    def __init__(self, aggr='add', norm = True, norm_type = 'A_hat_norm'):
        super(GraphConv, self).__init__(aggr=aggr)
        self.norm = norm
        self.norm_type = norm_type
    
    def forward(self, x, edge_index):
        if self.norm:
            norm, edge_index = self.get_norm(edge_index, x, self.norm_type)
        else:
            norm = torch.ones(edge_index.shape[1])
        aggr = self.propagate(edge_index, x=x, norm=norm)
        return aggr
    
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]. It contains the neighboring node features.
        # Each row contains the node feature of the destination node, i.e., if the edge is (i, j), it
        # contains the feature vector of node j.
        return norm.view(-1, 1) * x_j  # Each node feature in x_j is normalized according to its respective edge norm
    
    
    def get_norm(self, edge_index, x, norm_type ='A_hat_norm'):
        norm = None
        if norm_type == 'A_hat_norm':
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype) # contains the degree of each node. shape: (N)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
        elif norm_type == 'norm_laplacian':
            # Normalized Laplacian: I - D^(-0.5) * A * D^(-0.5)
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # Normalized Laplacian is (I - D^(-0.5) * A * D^(-0.5))
            norm = -deg_inv_sqrt[row] * deg_inv_sqrt[col]  # Negative sign for Laplacian
            # Adding self-loop correction (I part of I - A_hat)
            # Adding +1 for the self-loops (which would correspond to the identity matrix).
            self_loops = (row == col)
            norm[self_loops] = 1.0
        else:
            raise Exception('Norm not implemented yet!')
        return norm, edge_index


class KANConv(KANLayer):

    def __init__(self, in_channels, out_channels, k, g, grid_range, aggregate_first, aggr='add', norm=True,
                 scale_and_bias = False, norm_type = 'A_hat_norm'):
        super(KANConv, self).__init__(in_channels, out_channels, g, k, grid_range=grid_range, 
                                      scale_and_bias=scale_and_bias) 
        self.aggregate_first = aggregate_first
        self.graph_conv = GraphConv(aggr=aggr, norm=norm, norm_type=norm_type)
        
        
    def forward(self, X, edge_index, store_act=False):
        if self.aggregate_first:
            return super(KANConv, self).forward(self.graph_conv(X, edge_index), store_act)
        else:
            return self.graph_conv(super(KANConv, self).forward(X, store_act), edge_index)