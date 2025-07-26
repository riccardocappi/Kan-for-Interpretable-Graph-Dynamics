conda activate myenv
python post_processing_real_epid.py --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:0 --save_file=inferred_coeffs_h1n1.csv --model_type=GKAN &
sleep 3m
python post_processing_real_epid.py --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:0 --save_file=inferred_coeffs_h1n1.csv --model_type=MPNN &
sleep 1m
python post_processing_real_epid.py --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_sars.csv --model_type=GKAN &
sleep 3m
python post_processing_real_epid.py --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_sars.csv --model_type=MPNN &