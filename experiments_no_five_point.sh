conda activate myenv

# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=25 --study_name=biochemical_gkan_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=25 --study_name=biochemical_mpnn_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=25 --study_name=biochemical_llc_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=25 --study_name=biochemical_gkan_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=25 --study_name=biochemical_mpnn_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=25 --study_name=biochemical_llc_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=25 --study_name=biochemical_gkan_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=25 --study_name=biochemical_mpnn_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=25 --study_name=biochemical_llc_den_true_20db --process_id=0 --denoise --snr_db=20 &
# wait
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical.yml --method=optuna --n_trials=35 --study_name=biochemical_gkan_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_mpnn.yml --method=optuna --n_trials=35 --study_name=biochemical_mpnn_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_biochemical_llc.yml --method=optuna --n_trials=35 --study_name=biochemical_llc_no_fp_true --process_id=0 --deriv_method=finite_diff &
# wait

python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=25 --study_name=epidemics_gkan_den_true_70db_2 --process_id=0 --denoise --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=25 --study_name=epidemics_mpnn_den_true_70db_2 --process_id=0 --denoise --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=25 --study_name=epidemics_llc_den_true_70db_2 --process_id=0 --denoise --snr_db=70 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=25 --study_name=epidemics_gkan_den_true_50db_2 --process_id=0 --denoise --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=25 --study_name=epidemics_mpnn_den_true_50db_2 --process_id=0 --denoise --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=25 --study_name=epidemics_llc_den_true_50db_2 --process_id=0 --denoise --snr_db=50 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=25 --study_name=epidemics_gkan_den_true_20db_2 --process_id=0 --denoise --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=25 --study_name=epidemics_mpnn_den_true_20db_2 --process_id=0 --denoise --snr_db=20 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=25 --study_name=epidemics_llc_den_true_20db_2 --process_id=0 --denoise --snr_db=20 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics.yml --method=optuna --n_trials=35 --study_name=epidemics_gkan_no_fp_true_2 --process_id=0 --deriv_method=finite_diff &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_mpnn.yml --method=optuna --n_trials=35 --study_name=epidemics_mpnn_no_fp_true_2 --process_id=0 --deriv_method=finite_diff &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_epidemics_llc.yml --method=optuna --n_trials=35 --study_name=epidemics_llc_no_fp_true_2 --process_id=0 --deriv_method=finite_diff &
# wait

# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=25 --study_name=kuramoto_gkan_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=25 --study_name=kuramoto_mpnn_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=25 --study_name=kuramoto_llc_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=25 --study_name=kuramoto_gkan_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=25 --study_name=kuramoto_mpnn_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=25 --study_name=kuramoto_llc_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=25 --study_name=kuramoto_gkan_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=25 --study_name=kuramoto_mpnn_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=25 --study_name=kuramoto_llc_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=35 --study_name=kuramoto_gkan_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_mpnn.yml --method=optuna --n_trials=35 --study_name=kuramoto_mpnn_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto_llc.yml --method=optuna --n_trials=35 --study_name=kuramoto_llc_no_fp_true --process_id=0 --deriv_method=finite_diff &
# wait

# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=25 --study_name=population_gkan_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=25 --study_name=population_mpnn_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=25 --study_name=population_llc_den_true_70db --process_id=0 --denoise --snr_db=70 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=25 --study_name=population_gkan_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=25 --study_name=population_mpnn_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=25 --study_name=population_llc_den_true_50db --process_id=0 --denoise --snr_db=50 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=25 --study_name=population_gkan_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=25 --study_name=population_mpnn_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=25 --study_name=population_llc_den_true_20db --process_id=0 --denoise --snr_db=20 &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=35 --study_name=population_gkan_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_mpnn.yml --method=optuna --n_trials=35 --study_name=population_mpnn_no_fp_true --process_id=0 --deriv_method=finite_diff &
# sleep 1m
# python main.py --config=./configs/config_pred_deriv/config_ic1/config_population_llc.yml --method=optuna --n_trials=35 --study_name=population_llc_no_fp_true --process_id=0 --deriv_method=finite_diff &