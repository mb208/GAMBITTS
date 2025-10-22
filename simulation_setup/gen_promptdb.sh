#!/bin/bash
#SBATCH --account=tewaria0
#SBATCH --partition=spgpu
#SBATCH --job-name=promptdb
#SBATCH --output="gllogs/promptdb.out" 
#SBATCH --error="gllogs/promptdb.err" 
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-024:00:00

source /home/${USER}/.bashrc
conda activate nats 

DIMENSIONS=optimism,encouragement,clarity,severity,formality,threat,supportiveness,\
vision,politeness,humor,urgency,personalization,conciseness,authoritativeness,\
authenticity,complexity,emotiveness,actionability,detail,female-codedness

LLTYPE=qwen2.5:7b 

python3 -m scripts.create_prompt_db prompt_sim.dimension="[$DIMENSIONS]" \
                                    prompt_sim.llm_type="$LLTYPE" \
                                    prompt_sim.sim_path='data/sim_params/qwen2.5:7b_202507' \
                                    prompt_sim.model_prefix='../models/vae_model/' 
