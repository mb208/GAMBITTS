#!/bin/bash
#SBATCH --account=tewaria0
#SBATCH --partition=spgpu
#SBATCH --job-name=humor
#SBATCH --output="gllogs/humor.out" 
#SBATCH --error="gllogs/humor.err" 
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-024:00:00

source /home/${USER}/.bashrc
conda activate nats 


LLTYPE=qwen2.5:7b 
python -m  scripts.prompts4sim_patching prompt_sim.llm_type=$LLTYPE 
                               