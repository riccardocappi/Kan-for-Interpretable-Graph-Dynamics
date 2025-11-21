conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mlp.yml --method=optuna --n_trials=35 --study_name=epidemics_mlp_ode --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mlp.yml --method=optuna --n_trials=35 --study_name=biochemical_mlp_ode --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mlp.yml --method=optuna --n_trials=35 --study_name=population_mlp_ode --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mlp.yml --method=optuna --n_trials=35 --study_name=kuramoto_mlp_ode --process_id=0 &
