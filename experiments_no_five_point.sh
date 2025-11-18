conda activate myenv

python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_no_fp --process_id=0 --deriv_method=finite_diff &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=35 --study_name=epidemics_mpnn_no_fp --process_id=0 --deriv_method=finite_diff &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=35 --study_name=epidemics_llc_no_fp --process_id=0 --deriv_method=finite_diff &

