#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time 5:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=36
module load singularity
singularity exec  maze.sif python communication_easy/main.py --comp deigo --arg_name fi --curiosity hidden_state --delta 1  --agents $agents_per_job --previous_agents $previous_agents
