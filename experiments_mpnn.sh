conda activate myenv
nohup python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ --process_id=0 &
sleep 1m
python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ --process_id=1 &
sleep 1m
python main.py --config=./configs/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ --process_id=2 &
wait
echo "job done" > ./outputs/gkan-ode-mpnn.out &