#!/bin/bash -l
### job name
#SBATCH --job-name=GKAN-ODE-metrla

### 
#SBATCH --mail-user=riccardo.cappi@studenti.unipd.it

### Standard output and standard error for our job
#SBATCH --error=./slurm_files/gkanconv-ode-metrla.err
#SBATCH --output=./slurm_files/gkanconv-ode-metrla.out

### queue/partition choosed
#SBATCH --partition=testing
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  # Running three tasks in parallel

### RAM requirement
#SBATCH --mem=5G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=168:00:00

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
python main.py --config=./configs/config_metrla_mpnn.yml --method=optuna --n_trials=10 --study_name=metrla-ic-all-time --process_id=0 --storage=journal &
sleep 1m
python main.py --config=./configs/config_metrla_gkan.yml --method=optuna --n_trials=10 --study_name=metrla-ic-all-time --process_id=0 --storage=journal &
wait
echo "Finished job!"