import os


class ModelInterface():
    def __init__(self, model_path='./models'):
        super().__init__()
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        pass
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        pass
    