conda activate myenv
python main.py --config=./configs/config_ic5/config_kuramoto.yml --method=optuna --n_trials=10 --study_name=kuramoto_gkan_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_kuramoto.yml --method=optuna --n_trials=10 --study_name=kuramoto_gkan_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_kuramoto.yml --method=optuna --n_trials=10 --study_name=kuramoto_gkan_ic5_s5 --process_id=2 &
wait
python main.py --config=./configs/config_ic5/config_epidemics.yml --method=optuna --n_trials=10 --study_name=epidemics_gkan_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_epidemics.yml --method=optuna --n_trials=10 --study_name=epidemics_gkan_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_epidemics.yml --method=optuna --n_trials=10 --study_name=epidemics_gkan_ic5_s5 --process_id=2 &