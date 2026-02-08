#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:V100:1
    ## Type and the number of GPUs
    ## Don't change the GPU numbers.
    ## Follow https://rcic.uci.edu/hpc3/specs.html#specs
    ## to see all available GPUs.

#SBATCH --mem=30GB
#SBATCH --cpus-per-task=16

#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=sbatch_logs/slurm-%j.out
#SBATCH --error=sbatch_logs/slurm-%j.err

module load ffmpeg

mkdir -p sbatch_logs

# Virtual display for rgb_array rendering / video
srun xvfb-run -a python nikolaj_tests/base_test.py
