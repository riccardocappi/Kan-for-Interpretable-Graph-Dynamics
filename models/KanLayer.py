import torch
import math
import torch.nn.functional as F


class KANLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        scale_and_bias=False,
        compute_symbolic=False
    ):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.compute_symbolic = compute_symbolic

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.layer_mask = torch.nn.Parameter(torch.ones((out_features, in_features))).requires_grad_(False)
        self.symb_mask = torch.nn.Parameter(torch.zeros((out_features, in_features))).requires_grad_(False)
        self.symbolic_functions = [[lambda x: 0*x for _ in range(in_features)] for _ in range(out_features)]
        self.affine_params = torch.nn.Parameter(torch.rand((out_features, in_features, 4)))
        self.cache_act = None
        self.cache_preact = None
        self.symb_dict_names = {}
        
        self.layer_scale = torch.nn.Parameter(torch.ones(out_features)).requires_grad_(scale_and_bias)
        self.layer_bias = torch.nn.Parameter(torch.zeros(out_features)).requires_grad_(scale_and_bias)
        self.acts_scale_spline = None
        
        self.init_params()

    def init_params(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # Each basis function is define in the range [i, i + k + 1]
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        ).cpu()  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1).cpu()  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        if torch.cuda.is_available():
            solution = solution.cuda()

        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )


    def get_symbolic_output(self, x):
        # May be substituted with a sparse tensor for efficiency
        postacts = []
        for j in range(self.out_features):
            postacts_ = []
            for i in range(self.in_features):
                fn = self.symbolic_functions[j][i]
                a = self.affine_params[j,i,0]
                b = self.affine_params[j, i, 1]
                c = self.affine_params[j, i, 2]
                d = self.affine_params[j, i, 3]
                x_ji = c * fn(a * x[:, i] + b) + d
                postacts_.append(x_ji)
            postacts.append(torch.stack(postacts_, dim=1))
        return torch.stack(postacts, dim = 1)


    def get_activations(self, x):
        self.acts_scale_spline = []
        base_act = self.base_activation(x)
        base_output = base_act[:, None, :] * self.base_weight # (batch, out_features, in_features)

        splines = self.b_splines(x) # (batch, in, coeff)
        output_b_spline = torch.einsum('jik, bik -> bji', self.scaled_spline_weight, splines) # (batch, out_features, in_features)

        output_layer = output_b_spline + base_output
        
        if self.compute_symbolic:
            symb_output = self.get_symbolic_output(x) # (batch, out_features, in_features)
            output = self.layer_mask[None, :, :] * output_layer + self.symb_mask[None, :, :] * symb_output
        else:
            output = output_layer

        # Store acts
        self.cache_act = output.detach()
        self.cache_preact = x.detach()
        # For regularization loss
        input_range = torch.std(x, dim=0) + 1e-3
        output_range_spline = torch.std(output_layer, dim=0)
        self.acts_scale_spline = output_range_spline / input_range # (out_features, in_features)
        
        output = output.sum(dim=2) # (batch, out_features)
        return output


    def get_activations_efficient(self, x):
        original_shape = x.shape
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
    
    
    def forward(self, x: torch.Tensor, store_act=False):
        assert x.size(-1) == self.in_features
        x = x.reshape(-1, self.in_features)

        if store_act:
            output = self.get_activations(x) # (batch, out_features)
        else:
            output = self.get_activations_efficient(x) # (batch, out_features)
        
        # output = self.layer_scale[None,:] * output + self.layer_bias[None, :]
    
        return output
    
    
    def regularization_loss_fake(self, regularize_activation=1.0, regularize_entropy=1.0):
        
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        reg_loss = regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy
        return reg_loss, regularization_loss_activation, regularization_loss_entropy


    def regularization_loss_orig(self, mu_1=1.0, mu_2=1.0):
        assert self.acts_scale_spline is not None, 'Cannot use original L1 norm if activations are not saved'
        
        l1 = torch.sum(self.acts_scale_spline)
        p_row = self.acts_scale_spline / (torch.sum(self.acts_scale_spline, dim=1, keepdim=True) + 1)
        p_col = self.acts_scale_spline / (torch.sum(self.acts_scale_spline, dim=0, keepdim=True) + 1)
        entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
        entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
        entropy = (entropy_row + entropy_col)
        reg = mu_1*l1 + mu_2*entropy
        
        return reg, l1, entropy