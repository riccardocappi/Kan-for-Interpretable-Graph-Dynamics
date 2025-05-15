conda activate myenv
python main.py --config=./configs/config_ic5/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_biochemical_mpnn.yml --method=optuna --n_trials=15 --study_name=biochemical_mpnn_ic5_s5 --process_id=2 &
wait
echo "test" > ./outputs/test-ic5.out &
sleep 1m
python main.py --config=./configs/config_ic5/config_population_mpnn.yml --method=optuna --n_trials=15 --study_name=population_mpnn_ic5_s5 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_ic5/config_population_mpnn.yml --method=optuna --n_trials=15 --study_name=population_mpnn_ic5_s5 --process_id=1 &
sleep 1m
python main.py --config=./configs/config_ic5/config_population_mpnn.yml --method=optuna --n_trials=15 --study_name=population_mpnn_ic5_s5 --process_id=2 &