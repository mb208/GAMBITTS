#!/bin/bash
#SBATCH --account=tewaria1
#SBATCH --partition=spgpu
#SBATCH --job-name=promptdb
#SBATCH --output="gllogs/promptdb.out" 
#SBATCH --error="gllogs/promptdb.err" 
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00

source /home/${USER}/.bashrc
conda activate nats 


python3 -m scripts.create_prompt_db