conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_no_mult --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=35 --study_name=biochemical_gkan_no_mult --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=35 --study_name=population_gkan_no_mult --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=35 --study_name=kuramoto_gkan_no_mult --process_id=0 &
