#!/bin/bash

source /home/${USER}/.bashrc
conda activate nats 



# DIMENSIONS=optimism,encouragement,clarity,severity,formality,threat,supportiveness,\
# vision,politeness,humor,urgency,personalization,conciseness,authoritativeness,\
# authenticity,complexity,emotiveness,actionability,detail,female-codedness

DIMENSIONS=threat,humor
# NPROMPTS=10
#llama3.1
LLTYPE=qwen2.5:7b 
echo "Running prompt simulation with dimension: $DIMENSIONS, LLM type: $LLTYPE, number of prompts: $NPROMTPS"
python -m  scripts.prompts4sim --multirun prompt_sim.dimension=$DIMENSIONS \
                               prompt_sim.llm_type=$LLTYPE \
                               hydra.launcher.partition=spgpu \
                               hydra.launcher.account=tewaria0 \
                               hydra.launcher.gpus_per_node=1 \
                               hydra.launcher.mem_gb=24 \
                               hydra.launcher.timeout_min=1440 \
                               hydra.launcher.cpus_per_task=4 \
                               hydra.launcher.array_parallelism=4
