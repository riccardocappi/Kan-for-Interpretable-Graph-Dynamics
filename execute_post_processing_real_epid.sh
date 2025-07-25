conda activate myenv
python post_processing_real_epid.py --root=./data_real_epid_covid_int --device=cuda:0 --save_file=inferred_coeffs_covid.csv --model_type=GKAN &
sleep 3m
python post_processing_real_epid.py --root=./data_real_epid_covid_int --device=cuda:1 --save_file=inferred_coeffs_covid_.csv --model_type=MPNN &
