from torch.utils.data import Dataset

class SpatioTemporalGraphData(Dataset):
    def __init__(self, data, time):
        super().__init__()        
        self.data = data
        self.time = time
        
    
    def __len__(self):
        ics, n_sample, _, _ = self.data.shape
        return ics * (n_sample - 1)
    
    
    def __getitem__(self, index):
        index_ic = index // (self.data.size(1) - 1)
        index_sample = index % (self.data.size(1) - 1)
        
        x_in = self.data[index_ic, index_sample, : ,:]
        y_target = self.data[index_ic, index_sample + 1, :, :]
        t_start = self.time[index_ic, index_sample]
        t_target = self.time[index_ic, index_sample + 1]
        
        return x_in, y_target, t_start, t_target
        
        
        
        
        