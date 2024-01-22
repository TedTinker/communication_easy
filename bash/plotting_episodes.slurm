#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=50G
module load singularity
singularity exec  maze.sif python communication_easy/plotting_episodes.py --comp deigo --arg_title ___d+e+n+f+i+en+ef+ei+ni+fi+eni+efi___ --arg_name plotting_episodes
