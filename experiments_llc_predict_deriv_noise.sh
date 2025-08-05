conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=70 --study_name=epidemics_llc_70db_4 --process_id=0 --snr_db=70&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=70 --study_name=biochemical_llc_70db_4 --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=70 --study_name=population_llc_70db_4 --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=70 --study_name=kuramoto_llc_70db_4 --process_id=0 --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=70 --study_name=epidemics_llc_50db_4 --process_id=0 --snr_db=50&
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=70 --study_name=biochemical_llc_50db_4 --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=70 --study_name=population_llc_50db_4 --process_id=0 --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=70 --study_name=kuramoto_llc_50db_4 --process_id=0 --snr_db=50 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=70 --study_name=epidemics_llc_20db_4 --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=70 --study_name=biochemical_llc_20db_4 --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=70 --study_name=population_llc_20db_4 --process_id=0 --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=70 --study_name=kuramoto_llc_20db_4 --process_id=0 --snr_db=20 &