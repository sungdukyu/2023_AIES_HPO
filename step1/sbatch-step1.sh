#!/bin/bash
#SBATCH --job-name="Step_1"
#SBATCH --output="logs/srun-kerastuner-%j.%N.out"
#SBATCH --partition=GPU
#SBATCH --nodes=2
#SBATCH --gpus=16
#SBATCH --ntasks=17
#SBATCH --mem=192G
#SBATCH --export=ALL
#SBATCH --account=atm200007p
#SBATCH -t 00:30:00

srun --mpi=pmi2 --wait=0 bash run-dynamic.sh
