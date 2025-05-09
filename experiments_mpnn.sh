conda activate myenv
nohup  python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn_ --process_id=0 &
sleep 1m
python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn_ --process_id=1 &
sleep 1m
python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=10 --study_name=kuramoto-mpnn_ --process_id=2 &
wait

echo "job done" > ./outputs/gkan-ode-mpnn.out &