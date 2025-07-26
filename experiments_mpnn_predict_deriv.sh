conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=35 --study_name=epidemics_mpnn_ic1_s5_pd_mult_16 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=35 --study_name=biochemical_mpnn_ic1_s5_pd_mult_16 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=35 --study_name=population_mpnn_ic1_s5_pd_mult_16 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=35 --study_name=kuramoto_mpnn_ic1_s5_pd_mult_16 --process_id=0 &
