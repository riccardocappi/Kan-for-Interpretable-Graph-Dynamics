import os


class ModelInterface():
    """
    Defines the general structure that each implemented model must follow
    """
    def __init__(self, model_path='./models'):
        super().__init__()
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
    
    def regularization_loss(self, reg_loss_metrics:dict) -> float:
        """
        Computes the regularization loss (e.g. L1 norm of model's weights. Can be also 0. for non-KAN-based models)
        Args:
            -reg_loss_metrics : dictionary in which to save metrics related to the regularization loss (e.g. the entropy term of the KAN reg loss)
        
        Returns: regularization loss
        """
        pass
    
    
    def save_cached_data(self, dummy_x, dummy_edge_index):
        """
        This function is called in the post_processing step of Experiments, when saving model checkpoint. 
        Here you should save to file model's outputs and inputs that can be used later for symbolic regression.
        
        Args:
            dummy_x : Input for the forward pass of the model
            dummy_edge_index : Graph's edge_index for the forward pass of the model
        """
        pass
    
    def reset_params(self):
        """
        reset the parameters of the model. This function is called to reset model's weights after each run in the 
        objective function of the Experiments class.
        """
        pass
    