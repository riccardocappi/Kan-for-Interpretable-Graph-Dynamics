from torch.utils.data import Dataset

class GraphDynamics(Dataset):
    
    def __init__(self, data, time):
        super().__init__()
        self.data = data
        self.time = time
    

    def __len__(self):
        return self.data.size(1)

    
    def __getitem__(self, idx):
        return self.data[:, idx], self.time[:, idx]