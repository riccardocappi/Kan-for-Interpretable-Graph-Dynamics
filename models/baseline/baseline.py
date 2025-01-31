import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_path='./models'):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.model_path = model_path


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        return 0.0
    
    
    
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_path='./models', epsilon=0.):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        self.conv1 = GINConv(nn1, eps=epsilon, train_eps=False)
        
        self.model_path = model_path


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x
    
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        return 0.0