#!/bin/sh
echo "--- run-dynamic.sh ---"
echo SLURM_LOCALID $SLURM_LOCALID
echo SLURMD_NODENAME $SLURMD_NODENAME

conda activate tf
python keras-tuner-dynamic.p01.py > logs/keras-tuner-dynamic-$SLURM_JOBID-$SLURMD_NODENAME-$SLURM_LOCALID.log 2>&1

