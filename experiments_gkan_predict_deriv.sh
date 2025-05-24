conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic10/config_biochemical.yml --method=optuna --n_trials=30 --study_name=biochemical_gkan_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_neuronal.yml --method=optuna --n_trials=30 --study_name=neuronal_gkan_ic10_s5_pd_2 --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic10/config_epidemics.yml --method=optuna --n_trials=30 --study_name=epidemics_gkan_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic10_s5_pd_2 --process_id=0 &
