#!/bin/bash

source /home/${USER}/.bashrc
# source ${HOME}/.zshrc 
conda activate nats 

# Base ihs  environment 
# python -m scripts.ihs_linear_env --multirun \
#           env_setup=base_env_linear env_setup.filename=ihs_linear_outcome_3comps \
#           ~agent ~environment ~simulation

# # Varying N dims experiments

echo "Generating linear data generating environment configs"
python -m scripts.ihs_linear_env --multirun \
          env_setup=env_linear_3dims,env_linear_5dims,env_linear_10dims,env_linear_15dims,env_linear_full_dims \
          ~agent ~environment ~simulation \
          > ./logs/linear_mod_cfg_creation.txt 2>&1 &

echo "Generating neural net data generating environment configs"
python -m scripts.train_mlp --multirun \
          env_setup=env_nn_3dims,env_nn_5dims,env_nn_10dims,env_nn_15dims,env_nn_full_dims \
          ~agent ~environment ~simulation \
           > ./logs/nn_mod_cfg_creation.txt 2>&1 &