#!/bin/bash -l
### job name
#SBATCH --job-name=GKAN-ODE-pd

### 
#SBATCH --mail-user=riccardo.cappi@studenti.unipd.it

### Standard output and standard error for our job
#SBATCH --error=./slurm_files/gkanconv-ode-pd-3.err
#SBATCH --output=./slurm_files/gkanconv-ode-pd-3.out

### queue/partition choosed
#SBATCH --partition=testing
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3  # Running three tasks in parallel

### RAM requirement
#SBATCH --mem=3G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=72:00:00

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
cd /home/rcappi/Kan-for-Graph-Dyn
conda activate my_env
python main.py --config=./configs/config_pred_deriv/config_ic1/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic1_s5_pd_3 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic1/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic1_s5_pd_3 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic3/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic3_s5_pd_3 --process_id=0 &
wait
python main.py --config=./configs/config_pred_deriv/config_ic3/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic3_s5_pd_3 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_population.yml --method=optuna --n_trials=30 --study_name=population_gkan_ic5_s5_pd_3 --process_id=0 &
sleep 1m
python main.py --config=./configs/config_pred_deriv/config_ic5/config_kuramoto.yml --method=optuna --n_trials=30 --study_name=kuramoto_gkan_ic5_s5_pd_3 --process_id=0 &
wait

