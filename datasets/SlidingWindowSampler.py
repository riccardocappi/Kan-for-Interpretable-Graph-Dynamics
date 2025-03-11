from torch.utils.data import Sampler
import random

class SlidingWindowSampler(Sampler):
    def __init__(self, data_source, window_size, stride, shuffle=False):
        
        self.window_size = window_size
        self.shuffle = shuffle
        self.indices = list(range(0, len(data_source) - window_size + 1, stride))
        
        
    
    def __iter__(self):
        
        if self.shuffle:
            random.shuffle(self.indices)
        
        for i in self.indices:
            yield list(range(i, i + self.window_size))

    def __len__(self):
        return len(self.indices)
