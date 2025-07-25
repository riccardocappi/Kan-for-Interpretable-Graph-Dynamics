conda activate myenv
python post_processing_real_epid_gkan.py --root=./data_real_epid_covid_int --device=cuda:0 --save_file=inferred_coeffs_covid.csv &
sleep 1m
