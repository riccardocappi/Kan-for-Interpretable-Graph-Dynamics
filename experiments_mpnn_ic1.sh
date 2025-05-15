conda activate myenv
python main.py --config=./configs/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic1_s2 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic1_s2 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic1_s2 --process_id=2 &
wait
echo "job done" > ./outputs/test-ic1.out &