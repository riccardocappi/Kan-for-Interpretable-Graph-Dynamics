conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_ic1_s5_pd_mult_noise_70db --process_id=0 --snr_db=70&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=35 --study_name=biochemical_gkan_ic1_s5_pd_mult_noise_70db --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=35 --study_name=population_gkan_ic1_s5_pd_mult_noise_70db --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=35 --study_name=kuramoto_gkan_ic1_s5_pd_mult_noise_70db --process_id=0 --snr_db=70 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_ic1_s5_pd_mult_noise_50db --process_id=0 --snr_db=50&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=35 --study_name=biochemical_gkan_ic1_s5_pd_mult_noise_50db --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=35 --study_name=population_gkan_ic1_s5_pd_mult_noise_50db --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=35 --study_name=kuramoto_gkan_ic1_s5_pd_mult_noise_50db --process_id=0 --snr_db=50 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_ic1_s5_pd_mult_noise_20db --process_id=0 --snr_db=20&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=35 --study_name=biochemical_gkan_ic1_s5_pd_mult_noise_20db --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=35 --study_name=population_gkan_ic1_s5_pd_mult_noise_20db --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=35 --study_name=kuramoto_gkan_ic1_s5_pd_mult_noise_20db --process_id=0 --snr_db=20 &