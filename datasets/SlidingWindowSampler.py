from torch.utils.data import Sampler

class SlidingWindowSampler(Sampler):
    def __init__(self, data_source, window_size, stride):
        
        self.window_size = window_size
        self.indices = list(range(0, len(data_source) - window_size + 1, stride))
        
    
    def __iter__(self):
        for i in self.indices:
            yield self.indices[i:i+self.window_size]

    def __len__(self):
        return len(self.indices)
