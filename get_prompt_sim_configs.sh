#!/bin/bash

source /home/${USER}/.bashrc
# source ${HOME}/.zshrc 
conda activate nats 

# Generating simulation configs for varying prompt spaces according to difference 2nd stage variance
sd_lvl=_sd_0
for i in 3 5 10 15  
do
    if [ -z "$environments" ]; then
        environments="ihs_linear_outcome_${i}dims$sd_lvl,ihs_${i}dims_mlp_outcome_model$sd_lvl"
    else
        environments="${environments},ihs_linear_outcome_${i}dims$sd_lvl,ihs_${i}dims_mlp_outcome_model$sd_lvl"
    fi
done

environments="${environments},ihs_linear_outcome_full_dims$sd_lvl,ihs_full_dims_mlp_outcome_model$sd_lvl"
# environments=ihs_linear_outcome_3comps$sd_lvl,ihs_3comps_mlp_outcome_model$sd_lvl
# environments=report_outcome_model_base,report_outcome_model_3comps
# environments=report_outcome_model_3comps

echo "Using environments: $environments"
python -m scripts.gen_prompt_exp_cfg -m \
    environment=$environments \
    env_setup=nprompts_exp \
    env_setup.n_prompts=[5] \
    ~agent 