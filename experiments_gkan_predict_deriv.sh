conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic1_s5_pd_6 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic1_s5_pd_6 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic3_s5_pd_6 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic3_s5_pd_6 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic5_s5_pd_6 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic5_s5_pd_6 --process_id=0 &
