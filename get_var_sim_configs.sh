#!/bin/bash

source /home/${USER}/.bashrc
# source ${HOME}/.zshrc 
conda activate nats 

## 3 dim base environment 
environments=ihs_linear_outcome_3comps,ihs_3comps_mlp_outcome_model
python -m scripts.gen_variance_exp_cfg -m \
    env_setup=variance_exp \
    environment=ihs_linear_outcome_3comps,ihs_3comps_mlp_outcome_model \
    ~agent ~simulation

# 1 dim (optimism) base environment ,ihs_base_mlp_outcome_model
python -m scripts.gen_variance_exp_cfg -m \
    env_setup=variance_exp \
    environment=ihs_linear_outcome_base \
    ~agent ~simulation


# Iterating over varying dimensions for mlp and linear base environments
for i in 3 5 10 15  
do
    if [ -z "$environments" ]; then
        environments="ihs_linear_outcome_${i}dims,ihs_${i}dims_mlp_outcome_model"
    else
        environments="${environments},ihs_linear_outcome_${i}dims,ihs_${i}dims_mlp_outcome_model"
    fi
done

environments="${environments},ihs_linear_outcome_full_dims,ihs_full_dims_mlp_outcome_model"
env_setup="variance_exp_qwen"
echo "Generating environment configs with varying sd for the following: $environments"
echo "Using env_setup: $env_setup"
python -m scripts.gen_variance_exp_cfg -m \
    env_setup=$env_setup \
    environment=$environments \
    hydra.launcher.mem_gb=16 \
    ~agent ~simulation
