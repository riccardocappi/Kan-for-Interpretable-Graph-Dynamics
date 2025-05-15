conda activate myenv
python main.py --config=./configs/config_ic5/config_neuronal_mpnn.yml --method=optuna --n_trials=15 --study_name=neuronal_mpnn_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_neuronal_mpnn.yml --method=optuna --n_trials=15 --study_name=neuronal_mpnn_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_neuronal_mpnn.yml --method=optuna --n_trials=15 --study_name=neuronal_mpnn_ic5_s5 --process_id=2 &
wait
python main.py --config=./configs/config_ic5/config_kuramoto_mpnn.yml --method=optuna --n_trials=15 --study_name=kuramoto_mpnn_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_kuramoto_mpnn.yml --method=optuna --n_trials=15 --study_name=kuramoto_mpnn_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_kuramoto_mpnn.yml --method=optuna --n_trials=15 --study_name=kuramoto_mpnn_ic5_s5 --process_id=2 &
wait
python main.py --config=./configs/config_ic5/config_epidemics_mpnn.yml --method=optuna --n_trials=15 --study_name=epidemics_mpnn_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_epidemics_mpnn.yml --method=optuna --n_trials=15 --study_name=epidemics_mpnn_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_epidemics_mpnn.yml --method=optuna --n_trials=15 --study_name=epidemics_mpnn_ic5_s5 --process_id=2 &