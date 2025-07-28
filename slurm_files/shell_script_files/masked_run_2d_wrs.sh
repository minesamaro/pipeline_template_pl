#!/bin/bash

# Slurm job execution flags. REQUIRED TO SET: --job-name, --partition and --qos.
#SBATCH --job-name=wrsmasked
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out

init_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Initialization time: $init_time"

# Command-line to run the script for dataset generation.
srun python3 ../../src/scripts/run_experiment_pipeline.py --config-name main_resnet18_2d_one_class_masked_wrs
srun python3 ../../src/scripts/run_experiment_pipeline.py --config-name main_resnet50_2d_masked_wrs
srun python3 ../../src/scripts/run_experiment_pipeline.py --config-name main_vgg16_2d_masked_wrs

finish_time=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Finish time: $finish_time"

