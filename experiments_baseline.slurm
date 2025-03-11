#!/bin/bash -l
### job name
#SBATCH --job-name=GKAN-ODE-base-1

### 
#SBATCH --mail-user=riccardo.cappi@studenti.unipd.it

### Standard output and standard error for our job
#SBATCH --error=gkanconv-ode-baseline.err
#SBATCH --output=gkanconv-ode-baseline.out

### queue/partition choosed
#SBATCH --partition=testing

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=3G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=80:00:00

### GPU request
#SBATCH --gres=mps:30
#SBATCH --constraint=A5000


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
source .venv/gkanconv-ode/bin/activate
python main.py --config=./configs/config_kuramoto_mpnn.yml --method=optuna --n_trials=30 --study_name=barabasi-albert-mpnn-ic1 --process_id=1