import torch
import os
from models.utils.ODEBlock import ODEBlock
from utils.utils import save_black_box_to_file
from models.utils.MPNN import MPNN


class MLP_ODE(ODEBlock):

    def __init__(
        self,
        conv: MPNN,
        model_path='./models',
        adjoint=False,
        integration_method='dopri5',
        predict_deriv=False,
        all_t=False,
        **kwargs
    ):
        super().__init__(
            conv,
            model_path,
            adjoint,
            integration_method,
            predict_deriv=predict_deriv,
            all_t=all_t,
            **kwargs
        )

    def forward(self, snapshot):
        return super().forward(snapshot)

    def reset_params(self):
        """Reset all linear layers in the MLP."""
        for layer in self.conv.model.h_net:
            layer.reset_parameters()

    def regularization_loss(self, reg_loss_metrics):
        return 0.0

    def save_cached_data(self, dummy_x, dummy_edge_index, dummy_t, dummy_edge_attr):

        self.eval()

        # Enable caching
        self.conv.model.h_net.save_black_box = True

        with torch.no_grad():
            _ = self.conv.model.forward(dummy_x, dummy_edge_index, edge_attr=dummy_edge_attr, t=dummy_t)

        mlp_model_path = f"{self.model_path}/h_net"
        os.makedirs(mlp_model_path, exist_ok=True)

        save_black_box_to_file(
            folder_path=f"{mlp_model_path}/cached_data",
            cache_input=self.conv.model.h_net.cache_input,
            cache_output=self.conv.model.h_net.cache_output
        )
