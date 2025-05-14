#!/bin/bash

# Slurm job execution flags. REQUIRED TO SET: --job-name, --partition and --qos.
#SBATCH --job-name=test
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out

init_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Initialization time: $init_time"

# Command-line to run the script for dataset generation.
srun python3 ../../src/scripts/run_experiment_pipeline.py 

finish_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Finish time: $finish_time"

