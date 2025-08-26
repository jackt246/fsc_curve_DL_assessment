#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=4   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:2
#SBATCH --mem=100G   # memory per node
#SBATCH -J "fsc_DL"   # job name
#SBATCH -o "output.txt"   # job output file

# start GPU logging in the background
(
  while true; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> gpu_usage.log
      sleep 60
  done
) &

# save the PID so we can kill it later
LOGGER_PID=$!

export TORCH_HOME=/hps/nobackup/gerard/emdb/fsc_curve_DL_assessment/
python binarymodel.py

# stop the logging loop
kill $LOGGER_PID