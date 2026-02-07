#!/bin/bash
#SBATCH -A cs175_class    ## Account to charge
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=standard       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 8     ## Number of CPU cores
#SBATCH --output=sbatch_logs/slurm-%j.out   # Save standard output to logs/ folder
#SBATCH --error=sbatch_logs/slurm-%j.err    # Save standard error to logs/ folder (optional)

module load ffmpeg # necessary for saving gifs to tensorboard on HPC3
module load xvfb                 # Try loading this module if available

# The Critical Fix: Wrap python in xvfb-run
xvfb-run -a python nikolaj_tests/base_test.py
