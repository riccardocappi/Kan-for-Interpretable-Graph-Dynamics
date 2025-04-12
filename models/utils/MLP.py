import torch

class MLP(torch.nn.Module):
    """
    MLP Implementation
    """
    def __init__(self, hidden_layers, af, dropout_rate=0.0, save_black_box=False):
        super(MLP, self).__init__()
        self.af = af    # Activation function
        self.layers = torch.nn.ModuleList()
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