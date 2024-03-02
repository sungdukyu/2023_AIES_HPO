# Two-Step HPO (HyperParameter Optimization)


**Paper link: [Yu et al. (2023), "Two-step hyperparameter optimization method: Accelerating hyperparameter search by using a fraction of a training dataset"](https://doi.org/10.1175/AIES-D-23-0013.1)**



## [Example 1](https://github.com/sungdukyu/Two-step-HPO/tree/master/step1): Hyperparameter tuning for aersol activation emulators
The below code example to set up a two-step HPO using [Keras Tuner](https://keras.io/keras_tuner/) in a SLURM-managed HPC is taken from the supplemental material of Yu et al. (2023)

Keras Tuner provides a high-level interface that enables a distributed search mode with only four environmental variables. These include three Keras Tuner-specific variables (1-3) and one CUDA-related variable (4):

1. `KERASTUNER_TUNER_ID`: A unique ID assigned to manager and worker processes, with "chief" used for the manager process.
2. `KERASTUNER_ORACLE_IP`: IP address or hostname of the manager (chief) process.
3. `KERASTUNER_ORACLE_PORT`: Port number of the manager (chief) process.
4. `CUDA_VISIBLE_DEVICES`: Local GPU device ID to control the number and selection of GPUs assigned to a single process.

These four environmental variables are assigned dynamically depending on the availability of computing resources in an HPC. Accordingly, it can be technically challenging to write scripts that automatically set up correct values for the above four environmental variables. To address this, we provide code examples based on the scripts used in our case study.
 
Our case study was conducted on [PSC Bridges-2](https://www.psc.edu/resources/bridges-2/) HPC, which has 8 GPUs per node in its GPU partition. We used two nodes (16 GPUs in total) and assigned one GPU to each worker process. To automatically set the above four environmental variables, we wrote three scripts:

- [sbatch-keras-tuner.sh](https://github.com/sungdukyu/Two-step-HPO/blob/master/step1/sbatch-step1.sh)
: SLRUM job submission script that launches parallel jobs (in our case, ‘run-dynamic.sh’) using srun. The following SLURM options need to be set based on a user’s needs:

   - `partition`: name of a GPU partition
   - `nodes`: number GPU nodes
   - `gpus`: number of total GPUs processes, i.e., nodes x GPUs/node
   - `ntasks`: number of tasks, i.e., gpus +1. Extra one process accounts for the process for a manager. 

- [run-dynamic.sh](https://github.com/sungdukyu/Two-step-HPO/blob/master/step1/run-dynamic.sh)
: Intermediary script bridging the SLURM script (‘sbatch-kears-tuner.sh’) and the python script using Keras Tuner (‘step1-hpo-dynamic.py’). It is launched in parallel with each job step having its own SLURM-generated environmental variables, e.g., `SLURM_LOCALID` and `SLURMD_NODENAME`.
 
- [step1-hpo-dynamic.py](https://github.com/sungdukyu/Two-step-HPO/blob/master/step1/step1-hpo-dynamic.py)
: The python script for hyperparameter tuning using Keras Tuner, with the environmental variables (1-4) set automatically based on SLURM variables.:
   - `num_gpus_per_node` ([Line 31](https://github.com/sungdukyu/Two-step-HPO/blob/master/step1/step1-hpo-dynamic.py#L31)): the number of GPUs per node.


 
## [Example 2](https://github.com/sungdukyu/E3SM-MMF_baseline/tree/main/HPO/baseline_v1): Hyperparameter tuning for cloud-resolving model emulators
Example 1 assigns one whole GPU to one worker. However, depending on the size of neural networks or I/O burden, a given GPU might be under-utilized (that is, low GPU utilization rate). In such cases, multiple workers can be assigned to one GPU. We applied the two-step HPO for the MLP baseline of [ClimSim](https://arxiv.org/abs/2306.08754). We used [NERSC Perlmutter](https://www.nersc.gov/systems/perlmutter/) HPC and assigned 5 workers per GPU in this work. Two major changes from Example 1 are:

- `ntasks` in the [slurm job script](https://github.com/sungdukyu/E3SM-MMF_baseline/blob/main/HPO/baseline_v1/sbatch-kerastuner.gpu-shared.baseline_v1.sh) is now the number of workers plus one. In Perlmutter, each GPU node has 4 GPUs. So, using a single GPU node, ntasks = 5*4+1 (=21).
- `set_environment` function in the [python Keras Tuner script](https://github.com/sungdukyu/E3SM-MMF_baseline/blob/main/HPO/baseline_v1/hpo_baseline_v1.py#L17) now needs two arguments, `workers_per_node` and `workers_per_gpu`. 
