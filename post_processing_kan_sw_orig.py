import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from post_processing import get_test_set, post_process_gkan, get_symb_test_error
from utils.utils import load_config
import torch
import numpy as np
import random

def set_pytorch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    
    set_pytorch_seed(0)
    kur_config = load_config("./configs/config_pred_deriv/config_ic1/config_kuramoto.yml")

    KUR = get_test_set(
        dynamics=kur_config['name'],
        device='cuda',
        input_range=kur_config['input_range'],
        **kur_config['integration_kwargs']
    )

    g_symb = lambda x: torch.sin(x[:, 1] - x[:, 0]).unsqueeze(-1)
    h_symb = lambda x: 2.0 + 0.5 * x[:, 1].unsqueeze(-1)

    test_losses = get_symb_test_error(
        g_symb=g_symb,
        h_symb=h_symb,
        test_set=KUR,
        message_passing=True,
        include_time=False,
        is_symb=False
    )

    ts_mean = np.mean(test_losses)
    ts_var = np.var(test_losses)
    ts_std = np.std(test_losses)

    print(f"Mean Test loss of symbolic formula: {ts_mean}")
    print(f"Var Test loss of symbolic formula: {ts_var}")
    print(f"Std Test loss of symbolic formula: {ts_std}")

    """### Epidemics"""

    epid_config = load_config("./configs/config_pred_deriv/config_ic1/config_epidemics.yml")

    EPID = get_test_set(
        dynamics=epid_config['name'],
        device='cuda',
        input_range=epid_config['input_range'],
        **epid_config['integration_kwargs']
    )

    g_symb = lambda x: 0.5*x[:, 1].unsqueeze(-1) * (1 - x[:, 0].unsqueeze(-1))
    h_symb = lambda x: x[:, 1].unsqueeze(1) - 0.5 * x[:, 0].unsqueeze(-1)

    test_losses = get_symb_test_error(
        g_symb=g_symb,
        h_symb=h_symb,
        test_set=EPID,
        message_passing=True,
        include_time=False,
        is_symb=False
    )


    ts_mean = np.mean(test_losses)
    ts_var = np.var(test_losses)
    ts_std = np.std(test_losses)

    print(f"Mean Test loss of symbolic formula: {ts_mean}")
    print(f"Var Test loss of symbolic formula: {ts_var}")
    print(f"Std Test loss of symbolic formula: {ts_std}")

    """### Population"""

    pop_config = load_config("./configs/config_pred_deriv/config_ic1/config_population.yml")

    POP = get_test_set(
        dynamics=pop_config['name'],
        device='cuda',
        input_range=pop_config['input_range'],
        **pop_config['integration_kwargs']
    )

    g_symb = lambda x: 0.2*torch.pow(x[:, 1].unsqueeze(-1), 3)
    h_symb = lambda x: -0.5 * x[:, 0].unsqueeze(-1) + x[:, 1].unsqueeze(1)

    test_losses = get_symb_test_error(
        g_symb=g_symb,
        h_symb=h_symb,
        test_set=POP,
        message_passing=True,
        include_time=False,
        is_symb=False
    )

    ts_mean = np.mean(test_losses)
    ts_var = np.var(test_losses)
    ts_std = np.std(test_losses)

    print(f"Mean Test loss of symbolic formula: {ts_mean}")
    print(f"Var Test loss of symbolic formula: {ts_var}")
    print(f"Std Test loss of symbolic formula: {ts_std}")

    """### Biochemical"""

    bio_config = load_config("./configs/config_pred_deriv/config_ic1/config_biochemical.yml")

    BIO = get_test_set(
        dynamics=bio_config['name'],
        device='cuda',
        input_range=bio_config['input_range'],
        **bio_config['integration_kwargs']
    )

    g_symb = lambda x: (-0.5*x[:, 1] * x[:, 0]).unsqueeze(-1)
    h_symb = lambda x: (1.0 - 0.5 * x[:, 0]).unsqueeze(-1)  + x[:, 1].unsqueeze(-1)

    test_losses = get_symb_test_error(
        g_symb=g_symb,
        h_symb=h_symb,
        test_set=BIO,
        message_passing=True,
        include_time=False,
        is_symb=False
    )

    ts_mean = np.mean(test_losses)
    ts_var = np.var(test_losses)
    ts_std = np.std(test_losses)

    print(f"Mean Test loss of symbolic formula: {ts_mean}")
    print(f"Var Test loss of symbolic formula: {ts_var}")
    print(f"Std Test loss of symbolic formula: {ts_std}")    
    
    
    grid_orig = (
        [(-5, 5)],
        [0.01, 0.05, 0.1],
        [1e-5, 0.3, 0.7, 0.9]
    )
    
    # BIO ------------------------------------------------------------------
    post_process_gkan(
        config=bio_config,
        model_path="./saved_models_optuna/model-biochemical-gkan/biochemical_gkan_ic1_s5_pd_mult_12/0",
        test_set=BIO,
        device='cuda',
        n_g_hidden_layers=2,
        n_h_hidden_layers=2,
        sample_size=10000,
        message_passing=False,
        include_time=False,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
        compute_mult=True,
        grid_orig=grid_orig,
        skip_bb=True,
        res_file_name="sw_orig_kan_final_2.json"
    )
    
    model_paths_gkan = [
        "./saved_models_optuna/model-biochemical-gkan/biochemical_gkan_ic1_s5_pd_mult_noise_70db_2/0",
        "./saved_models_optuna/model-biochemical-gkan/biochemical_gkan_ic1_s5_pd_mult_noise_50db_2/0",
        "./saved_models_optuna/model-biochemical-gkan/biochemical_gkan_ic1_s5_pd_mult_noise_20db_2/0"
    ]

    for model_path in model_paths_gkan:
        print(model_path)

        post_process_gkan(
            config=bio_config,
            model_path=model_path,
            test_set=BIO,
            device='cuda',
            n_g_hidden_layers=2,
            n_h_hidden_layers=2,
            sample_size=10000,
            message_passing=False,
            include_time=False,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
            eval_model=True,
            compute_mult=True,
            grid_orig=grid_orig,
            skip_bb=True,
            res_file_name="sw_orig_kan_final_2.json"
        )
        # ------------------------------------------------------------------
        
    # KUR ------------------------------------------------------------------
    post_process_gkan(
        config=kur_config,
        model_path="./saved_models_optuna/model-kuramoto-gkan/kuramoto_gkan_ic1_s5_pd_mult_12/0",
        test_set=KUR,
        device='cuda',
        n_g_hidden_layers=2,
        n_h_hidden_layers=2,
        sample_size=10000,
        message_passing=False,
        include_time=False,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
        compute_mult=True,
        grid_orig=grid_orig,
        skip_bb=True,
        res_file_name="sw_orig_kan_final_2.json"
    )
    
    model_paths_gkan = [
        "./saved_models_optuna/model-kuramoto-gkan/kuramoto_gkan_ic1_s5_pd_mult_noise_70db_2/0",
        "./saved_models_optuna/model-kuramoto-gkan/kuramoto_gkan_ic1_s5_pd_mult_noise_50db_2/0",
        "./saved_models_optuna/model-kuramoto-gkan/kuramoto_gkan_ic1_s5_pd_mult_noise_20db/0",
    ]

    for model_path in model_paths_gkan:
        print(model_path)

        post_process_gkan(
            config=kur_config,
            model_path=model_path,
            test_set=KUR,
            device='cuda',
            n_g_hidden_layers=2,
            n_h_hidden_layers=2,
            sample_size=10000,
            message_passing=False,
            include_time=False,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
            eval_model=True,
            compute_mult=True,
            grid_orig=grid_orig,
            skip_bb=True,
            res_file_name="sw_orig_kan_final_2.json"
        )
        
    # ------------------------------------------------------------------
    
    # EPID ------------------------------------------------------------------
    post_process_gkan(
        config=epid_config,
        model_path="./saved_models_optuna/model-epidemics-gkan/epidemics_gkan_ic1_s5_pd_mult_12/0",
        test_set=EPID,
        device='cuda',
        n_g_hidden_layers=2,
        n_h_hidden_layers=2,
        sample_size=10000,
        message_passing=False,
        include_time=False,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
        compute_mult=True,
        grid_orig=grid_orig,
        skip_bb=True,
        res_file_name="sw_orig_kan_final_2.json"
    )
    
    model_paths_gkan = [
        "./saved_models_optuna/model-epidemics-gkan/epidemics_gkan_ic1_s5_pd_mult_noise_70db_2/0",
        "./saved_models_optuna/model-epidemics-gkan/epidemics_gkan_ic1_s5_pd_mult_noise_50db_2/0",
        "./saved_models_optuna/model-epidemics-gkan/epidemics_gkan_ic1_s5_pd_mult_noise_20db_2/0",
    ]

    for model_path in model_paths_gkan:
        print(model_path)
        post_process_gkan(
            config=epid_config,
            model_path=model_path,
            test_set=EPID,
            device='cuda',
            n_g_hidden_layers=2,
            n_h_hidden_layers=2,
            sample_size=10000,
            message_passing=False,
            include_time=False,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
            eval_model=True,
            compute_mult=True,
            grid_orig=grid_orig,
            skip_bb=True,
            res_file_name="sw_orig_kan_final_2.json"
        )
    
    # ------------------------------------------------------------------
    
    # POP ------------------------------------------------------------------
    post_process_gkan(
        config=pop_config,
        model_path="./saved_models_optuna/model-population-gkan/population_gkan_ic1_s5_pd_mult_12/0",
        test_set=POP,
        device='cuda',
        n_g_hidden_layers=2,
        n_h_hidden_layers=2,
        sample_size=10000,
        message_passing=False,
        include_time=False,
        atol=1e-5,
        rtol=1e-5,
        method="dopri5",
        compute_mult=True,
        grid_orig=grid_orig,
        skip_bb=True,
        res_file_name="sw_orig_kan_final_2.json"
    )
    
    model_paths_gkan = [
        "./saved_models_optuna/model-population-gkan/population_gkan_ic1_s5_pd_mult_noise_70db_2/0",
        "./saved_models_optuna/model-population-gkan/population_gkan_ic1_s5_pd_mult_noise_50db_2/0",
        "./saved_models_optuna/model-population-gkan/population_gkan_ic1_s5_pd_mult_noise_20db_2/0",
    ]

    for model_path in model_paths_gkan:
        print(model_path)
        post_process_gkan(
            config=pop_config,
            model_path=model_path,
            test_set=POP,
            device='cuda',
            n_g_hidden_layers=2,
            n_h_hidden_layers=2,
            sample_size=10000,
            message_passing=False,
            include_time=False,
            atol=1e-5,
            rtol=1e-5,
            method="dopri5",
            eval_model=True,
            compute_mult=True,
            grid_orig=grid_orig,
            skip_bb=True,
            res_file_name="sw_orig_kan_final_2.json"
        ) 
    
    # ------------------------------------------------------------------
    
    