#!/bin/sh
echo "--- run-dynamic.sh ---"
echo SLURM_LOCALID $SLURM_LOCALID
echo SLURMD_NODENAME $SLURMD_NODENAME

conda activate tf
python step1-hpo-dynamic.py > logs/step1-hpo-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1

