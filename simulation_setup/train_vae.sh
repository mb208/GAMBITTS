#!/bin/bash

#### Below is from when this was standalone sbatch script, but now this is as an argument to another script that runs the 
#### slurm job.

# ######SBATCH --account=tewaria1
# ######SBATCH --partition=spgpu
# ######SBATCH --job-name=clarity_vae
# ######SBATCH --output="gllogs/clarity_2000.out" 
# ######SBATCH --error="gllogs/clarity_2000.err" 
# ######SBATCH -N 1
# ######SBATCH --gpus-per-node=1
# ######SBATCH --cpus-per-task=4
# ######SBATCH --mem=16G
# ######SBATCH -t 0-01:00:00

source /home/${USER}/.bashrc
conda activate nats 

python -m  scripts.train_vae vae.style_dim=$1 vae.data_folder=$2 vae.filename=$3