conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=35 --study_name=population_mpnn_ic1_s5_pd_14 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=35 --study_name=kuramoto_mpnn_ic1_s5_pd_14 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_population_mpnn.yml --method=optuna --n_trials=35 --study_name=population_mpnn_ic3_s5_pd_14 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_kuramoto_mpnn.yml --method=optuna --n_trials=35 --study_name=kuramoto_mpnn_ic3_s5_pd_14 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_population_mpnn.yml --method=optuna --n_trials=35 --study_name=population_mpnn_ic5_s5_pd_14 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_kuramoto_mpnn.yml --method=optuna --n_trials=35 --study_name=kuramoto_mpnn_ic5_s5_pd_14 --process_id=0 &
