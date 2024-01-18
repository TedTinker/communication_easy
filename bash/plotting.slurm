#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=50G
module load singularity
singularity exec  maze.sif python communication_easy/plotting.py --comp deigo --arg_title ___d___ --arg_name plotting
