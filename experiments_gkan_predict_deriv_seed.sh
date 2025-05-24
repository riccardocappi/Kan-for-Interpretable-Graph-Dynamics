conda activate myenv
python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=30 --study_name=biochemical_gkan_ic1_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic1_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_neuronal.yml --method=optuna --n_trials=30 --study_name=neuronal_gkan_ic1_s5_pd_seed --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=30 --study_name=epidemics_gkan_ic1_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic1_s5_pd_seed --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic3/config_biochemical.yml --method=optuna --n_trials=30 --study_name=biochemical_gkan_ic3_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic3_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_neuronal.yml --method=optuna --n_trials=30 --study_name=neuronal_gkan_ic3_s5_pd_seed --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic3/config_epidemics.yml --method=optuna --n_trials=30 --study_name=epidemics_gkan_ic3_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic3_s5_pd_seed --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic5/config_biochemical.yml --method=optuna --n_trials=30 --study_name=biochemical_gkan_ic5_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic5_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_neuronal.yml --method=optuna --n_trials=30 --study_name=neuronal_gkan_ic5_s5_pd_seed --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic5/config_epidemics.yml --method=optuna --n_trials=30 --study_name=epidemics_gkan_ic5_s5_pd_seed --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic5_s5_pd_seed --process_id=0 &
wait