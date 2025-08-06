conda activate myenv
python post_processing_real_epid.py  --scale --root=./data_real_epid_covid_int --infection_data=./data/RealEpidemics/infected_numbers_covid.csv --inf_threshold=500 --device=cuda:0 --save_file=inferred_coeffs_covid_ts_correct.csv --model_type=GKAN &
sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_covid_int --infection_data=./data/RealEpidemics/infected_numbers_covid.csv --inf_threshold=500 --device=cuda:0 --save_file=inferred_coeffs_covid_ts.csv --model_type=MPNN &
# sleep 1m
python post_processing_real_epid.py  --scale --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_h1n1_ts_correct.csv --model_type=GKAN &
sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_h1n1_ts.csv --model_type=MPNN &
# sleep 1m
python post_processing_real_epid.py  --scale --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:2 --save_file=inferred_coeffs_sars_ts_correct.csv --model_type=GKAN &
sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:3 --save_file=inferred_coeffs_sars_ts.csv --model_type=MPNN &
# sleep 1m
# python post_processing_real_epid.py  --root=./data_real_epid_covid_int --infection_data=./data/RealEpidemics/infected_numbers_covid.csv --inf_threshold=500 --device=cuda:0 --save_file=inferred_coeffs_covid_new.csv --model_type=TSS > ./outputs/output_tss_covid.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_h1n1_new.csv --model_type=TSS > ./outputs/output_tss_h1n1.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:2 --save_file=inferred_coeffs_sars_new.csv --model_type=TSS > ./outputs/output_tss_sars.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_covid_int --infection_data=./data/RealEpidemics/infected_numbers_covid.csv --inf_threshold=500 --device=cuda:0 --save_file=inferred_coeffs_covid_new.csv --model_type=LLC > ./outputs/output_llc_covid.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_h1n1_new.csv --model_type=LLC > ./outputs/output_llc_h1n1.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:2 --save_file=inferred_coeffs_sars_new.csv --model_type=LLC > ./outputs/output_llc_sars.txt 2>&1 &

# python post_processing_real_epid.py  --scale --root=./data_real_epid_covid_int --infection_data=./data/RealEpidemics/infected_numbers_covid.csv --inf_threshold=500 --device=cuda:0 --save_file=inferred_coeffs_covid_sw.csv --model_type=SW > ./outputs/output_sw_covid.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_h1n1_int --infection_data=./data/RealEpidemics/infected_numbers_H1N1.csv --inf_threshold=100 --device=cuda:1 --save_file=inferred_coeffs_h1n1_sw.csv --model_type=SW > ./outputs/output_sw_h1n1.txt 2>&1 &
# sleep 1m
# python post_processing_real_epid.py  --scale --root=./data_real_epid_sars_int --infection_data=./data/RealEpidemics/infected_numbers_sars.csv --inf_threshold=100 --device=cuda:2 --save_file=inferred_coeffs_sars_sw.csv --model_type=SW > ./outputs/output_sw_sars.txt 2>&1 &
