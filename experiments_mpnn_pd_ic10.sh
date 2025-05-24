conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic10/config_biochemical_mpnn.yml --method=optuna --n_trials=30 --study_name=biochemical_mpnn_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_population_mpnn.yml --method=optuna --n_trials=30 --study_name=population_mpnn_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_neuronal_mpnn.yml --method=optuna --n_trials=30 --study_name=neuronal_mpnn_ic10_s5_pd_2 --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic10/config_epidemics_mpnn.yml --method=optuna --n_trials=30 --study_name=epidemics_mpnn_ic10_s5_pd_2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic10/config_kuramoto_mpnn.yml --method=optuna --n_trials=30 --study_name=kuramoto_mpnn_ic10_s5_pd_2 --process_id=0 &
