import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import random
import numpy as np

def set_pytorch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

set_pytorch_seed(0)

import torch
import sympytorch
import sympy as sp
from datasets.RealEpidemics import RealEpidemics

import numpy as np
from post_processing import get_model



from post_processing_real_epid_gkan import fit_param_per_country_gd, get_scaler


if __name__ == "__main__":
    

    real_epid_data = RealEpidemics(
        root = './data_real_epid_covid_int',
        name = 'RealEpid',
        predict_deriv=False,
        history=1,
        horizon=44,
        scale=False
    )
    
    import json

    with open('./data_real_epid_covid_int/RealEpid/countries_dict.json', 'r') as f:
        countries_dict = json.load(f)
        
    class symb_wrapper(torch.nn.Module):
        def __init__(self, symb, is_self_interaction = True):
            super().__init__()
            self.symb = symb
            self.is_self_interaction = is_self_interaction
        
        def forward(self, x):
            if self.is_self_interaction:
                return self.symb(x_i=x[:, 0])
            else:
                return self.symb(x_i=x[:, 0], x_j=x[:, 1])
            
    model_path = "./saved_models_optuna/model-real-epid-mpnn/real_epid_mpnn_7/0"

    # import random

    def build_symb_mpnn_to_opt():
        x_i, x_j = sp.symbols('x_i x_j')

        a = 3.7716758
        b = 1.9867662
        c = 1.2657967

        eps = 1e-3
        expr1 = sp.ln(sp.Max(sp.tan(x_i + c)**2 + 1, eps))
        expr2 = a * sp.ln(sp.Max(x_i + b, eps))

        g_symb = sympytorch.SymPyModule(expressions=[expr1])
        h_symb = sympytorch.SymPyModule(expressions=[expr2])
        
        g_symb = symb_wrapper(g_symb, is_self_interaction=False)
        h_symb = symb_wrapper(h_symb, is_self_interaction=True)

        symb_model = get_model(
            g = g_symb,
            h = h_symb,
            message_passing=False,
            include_time=False,
            integration_method='rk4',
            eval=False,
            all_t=True
        )

        symb_model = symb_model.train()   
        symb_model = symb_model.to('cuda:1')
        
        return symb_model


    fine_tune_data = RealEpidemics(
        root = './data_real_epid_covid_ft_scaled',
        name = 'RealEpid',
        predict_deriv=False,
        history=1,
        horizon=9,
        stride=7,
        scale=True,
        scale_range=(-1, 1),
        train_perc=0.8
    )

    tr_len = len(fine_tune_data)
    tr_end = int(0.8 * tr_len)
    train_set = fine_tune_data[:tr_end]
    valid_set = fine_tune_data[tr_end:]

    fit_param_per_country_gd(
        data=train_set,
        valid_data=valid_set,
        build_symb_model=build_symb_mpnn_to_opt,
        countries_dict=countries_dict,
        model_path=model_path,
        patience=30,
        log=10,
        scaler=None,
        tr_perc=0.8,
        save_file="prova_2.csv",
        optimizer_type="adam",
        device="cuda:1"
    )

        