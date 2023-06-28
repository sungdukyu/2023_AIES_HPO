#!/bin/bash
#SBATCH --job-name="Step_2"
#SBATCH --output="logs/step-2.%j.out"
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH -t 03:00:00

conda activate tf
f_json='../step1/results/P05/trial_00023/trial.json'
f_trained_model='./top_models/P05-trial_00023.hdf5'
python step2-retrain-trials.py ${f_json} ${f_trained_model} 
