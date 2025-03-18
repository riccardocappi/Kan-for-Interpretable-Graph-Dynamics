# GKAN-ODE

## Code organization
The pipeline logic is described by the `Experiments` class and it includes three components:
- Pre-processing.
- Model selection with `optuna`.
- Saving best model checkpoint to file

`Experiments` is an abstract class and defines some abstract methods that each specific experiment class must implement. In particylar, each sub-class of `Experiments` must specify how to pre process data and how to construct the current model, given an `optuna` trial. For example, the `ExperimentsGKAN` class defines the experiments pipeline for the GKAN-ODE models. It implements the `get_model_opt` abstract method by constructing a GKAN-ODE model with the parameters returned by the current optuna trial.
```
def get_model_opt(self, trial):
    ...
    g_net = KAN(**g_net_config)
    h_net = KAN(**h_net_config)

    net = GKAN_ODE(
        h_net=h_net,
        g_net=g_net,
        model_path = f"{self.model_path}/gkan",
        device = self.device,
        lmbd_g=lamb_g,
        lmbd_h=lamb_h,
        message_passing=self.config.get("message_passing", True)
    )
    
    model = NetWrapper(net, self.edge_index, update_grid=False)
    model = model.to(torch.device(self.device))
    
    return model
```
Note that the returned model must be instance of `NetWrapper`. This class just wraps around a torch module in order to properly work with the `torchdiffeq` methods. Actually, the input model for `NetWrapper` should be also an instance of `ModelInterface`. In fact, the `GKAN_ODE` class extends both `torch_geometric.nn.MessagePassing` and `model.utils.ModelInterface`.
```
class GKAN_ODE(MessagePassing, ModelInterface):
```
`ModelInterface` provides the general structure that models must follow in order to properly work with the Experiments pipeline. In particular, each model must implement the following three methods:
- regularization_loss: Computes the regularization loss (e.g. L1 norm of model's weights. Can be also 0. for non-KAN-based models). This function is called inside the `fit` method inside `train_and_eval.py`.

- save_cached_data: This function is called in the post_processing step of `Experiments`, when saving model checkpoint. Here you should save to file model's outputs and inputs that can be used later for symbolic regression. 

- reset_params: reset the parameters of the model. This function is called to reset model's weights after each run in the `objective` function of the `Experiments` class.

The training process is defined in the 