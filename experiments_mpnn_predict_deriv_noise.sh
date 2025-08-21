conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=70 --study_name=epidemics_mpnn_ic1_s5_pd_mult_noise_70db_2 --process_id=0 --snr_db=70&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=70 --study_name=biochemical_mpnn_ic1_s5_pd_mult_noise_70db_2 --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=70 --study_name=population_mpnn_ic1_s5_pd_mult_noise_70db_2 --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=70 --study_name=kuramoto_mpnn_ic1_s5_pd_mult_noise_70db_2 --process_id=0 --snr_db=70 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=70 --study_name=epidemics_mpnn_ic1_s5_pd_mult_noise_50db_2 --process_id=0 --snr_db=50&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=70 --study_name=biochemical_mpnn_ic1_s5_pd_mult_noise_50db_2 --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=70 --study_name=population_mpnn_ic1_s5_pd_mult_noise_50db_2 --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=70 --study_name=kuramoto_mpnn_ic1_s5_pd_mult_noise_50db_2 --process_id=0 --snr_db=50 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=70 --study_name=epidemics_mpnn_ic1_s5_pd_mult_noise_20db_2 --process_id=0 --snr_db=20&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=70 --study_name=biochemical_mpnn_ic1_s5_pd_mult_noise_20db_2 --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=70 --study_name=population_mpnn_ic1_s5_pd_mult_noise_20db_2 --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=70 --study_name=kuramoto_mpnn_ic1_s5_pd_mult_noise_20db_2 --process_id=0 --snr_db=20 &