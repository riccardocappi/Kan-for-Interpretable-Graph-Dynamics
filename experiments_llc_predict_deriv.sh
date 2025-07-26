conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=35 --study_name=epidemics_llc --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=35 --study_name=biochemical_llc --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=35 --study_name=population_llc --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=35 --study_name=kuramoto_llc --process_id=0 &