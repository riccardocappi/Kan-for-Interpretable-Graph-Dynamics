#!/bin/bash -l
### job name
#SBATCH --job-name=GKAN-ODE-in-noise

### 
#SBATCH --mail-user=riccardo.cappi@studenti.unipd.it

### Standard output and standard error for our job
#SBATCH --error=./slurm_files/gkanconv-ode-in-noise.err
#SBATCH --output=./slurm_files/gkanconv-ode-in-noise.out

### queue/partition choosed
#SBATCH --partition=testing
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3  # Running three tasks in parallel

### RAM requirement
#SBATCH --mem=3G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=180:00:00

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=A6000


### Some useful informative commands
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on the following nodes: '
echo ${SLURM_NODELIST}
echo

SHELL=/bin/bash

### Jobs execution commands
cd /home/rcappi/Kan-for-Graph-Dyn
conda activate my_env

python main.py --config=./configs/config_input_noise/config_kuramoto_input_noise.yml --method=optuna --n_trials=15 --study_name=kuramoto-gkan_in_noise_ --process_id=0 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_kuramoto_input_noise.yml --method=optuna --n_trials=15 --study_name=kuramoto-gkan_in_noise_ --process_id=1 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_kuramoto_input_noise.yml --method=optuna --n_trials=15 --study_name=kuramoto-gkan_in_noise_ --process_id=2 &
wait
python main.py --config=./configs/config_input_noise/config_epidemics_input_noise.yml --method=optuna --n_trials=15 --study_name=epidemics_gkan_in_noise --process_id=0 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_epidemics_input_noise.yml --method=optuna --n_trials=15 --study_name=epidemics_gkan_in_noise --process_id=1 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_epidemics_input_noise.yml --method=optuna --n_trials=15 --study_name=epidemics_gkan_in_noise --process_id=2 &
wait
python main.py --config=./configs/config_input_noise/config_neuronal_input_noise.yml --method=optuna --n_trials=15 --study_name=neuronal_gkan_in_noise --process_id=0 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_neuronal_input_noise.yml --method=optuna --n_trials=15 --study_name=neuronal_gkan_in_noise --process_id=1 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_neuronal_input_noise.yml --method=optuna --n_trials=15 --study_name=neuronal_gkan_in_noise --process_id=2 &
wait
python main.py --config=./configs/config_input_noise/config_biochemical_input_noise.yml --method=optuna --n_trials=15 --study_name=biochemical_gkan_in_noise --process_id=0 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_biochemical_input_noise.yml --method=optuna --n_trials=15 --study_name=biochemical_gkan_in_noise --process_id=1 &
sleep 1m
python main.py --config=./configs/config_input_noise/config_biochemical_input_noise.yml --method=optuna --n_trials=15 --study_name=biochemical_gkan_in_noise --process_id=2 &
wait
echo "Finished"