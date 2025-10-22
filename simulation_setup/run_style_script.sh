#!/bin/bash
#SBATCH --account=tewaria0
#SBATCH --partition=spgpu
#SBATCH --job-name=text_sim_style
#SBATCH --output="gllogs/text_sim_style.out" 
#SBATCH --error="gllogs/text_sim_style.err" 
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-024:00:00


source /home/${USER}/.bashrc
conda activate nats 

DIMENSIONS=optimism
LLTYPE=qwen2.5:7b 

python -m scripts.generate_interventions text_sim.style_dim=$DIMENSIONS text_sim.llm_type=$LLTYPE
