nohup bash -c '
conda activate myenv &&

python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn --process_id=0 &
sleep 1m
python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn --process_id=1 &
sleep 1m
python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn --process_id=2 &
wait

python main.py --config=./configs/config_epidemics_mpnn.yml --method=optuna --n_trials=10 --study_name=epidemics-mpnn --process_id=0 &
sleep 1m
python main.py --config=./configs/config_epidemics_mpnn.yml --method=optuna --n_trials=10 --study_name=epidemics-mpnn --process_id=1 &
sleep 1m
python main.py --config=./configs/config_epidemics_mpnn.yml --method=optuna --n_trials=10 --study_name=epidemics-mpnn --process_id=2 &
wait

python main.py --config=./configs/config_neuronal_mpnn.yml --method=optuna --n_trials=10 --study_name=neuronal-mpnn --process_id=0 &
sleep 1m
python main.py --config=./configs/config_neuronal_mpnn.yml --method=optuna --n_trials=10 --study_name=neuronal-mpnn --process_id=1 &
sleep 1m
python main.py --config=./configs/config_neuronal_mpnn.yml --method=optuna --n_trials=10 --study_name=neuronal-mpnn --process_id=2 &
wait

python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=10 --study_name=biochemical-mpnn --process_id=0 &
sleep 1m
python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=10 --study_name=biochemical-mpnn --process_id=1 &
sleep 1m
python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=10 --study_name=biochemical-mpnn --process_id=2 &
wait

echo "job done"
' > ./outputs/gkan-ode-mpnn.out 2>&1 &