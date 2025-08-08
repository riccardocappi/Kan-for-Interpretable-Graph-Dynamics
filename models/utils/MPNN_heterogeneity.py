import torch
from torch_geometric.nn import MessagePassing

class ODEGraphLayer(MessagePassing):
    def __init__(self, a, b, device='cuda'):
        super().__init__(aggr="add")  # sum aggregation
        self.a = torch.from_numpy(a).to(device)
        self.b = torch.from_numpy(b).to(device)

    def forward(self, x, edge_index, edge_attr, t):

        # propagate messages from neighbors
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        out = out + self.a * x
        return out

    def message(self, x_i, x_j, edge_attr):
        mes =  (1 / 1 + torch.exp(- (x_j - x_i)))
        out = mes if edge_attr is None else edge_attr.view(-1, 1) * mes
        return out
    
    def update(self, aggr_out, x):
        out = self.b * aggr_out
        return out
