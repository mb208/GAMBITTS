#!/bin/sh
source /home/${USER}/.bashrc
# source ${HOME}/.zshrc 
conda activate nats 



############################################################################################################
######
######
######      Returns configuration for Environemnt Generarative model trained on IHS data according to specs
######
######
############################################################################################################

##### Linear Environment Outcome Models

## Single Component - Optimism Only
python -m scripts.ihs_linear_env env_setup=base_env_linear \
        env_setup.dimensions=[optimism] env_setup.filename=ihs_linear_outcome_base env_setup.dim_batch=batch_0 

## Three Components - Optimism, Severity, Formality
python -m scripts.ihs_linear_env env_setup=base_env_linear \
        env_setup.dimensions=[optimism,severity,formality] \
        env_setup.filename=ihs_linear_outcome_3comps \
        env_setup.dim_batch=batch_1

##### MLP Environment Outcome Models
## Single Component - Optimism Only
python -m scripts.train_mlp -m env_setup=base_env_nn \
        env_setup.dimensions=[optimism] env_setup.model_id=ihs_base env_setup.dim_batch=batch_0

## Three Components - Optimism, Severity, Formality
python -m scripts.train_mlp -m env_setup=base_env_nn \
        env_setup.dimensions=[optimism,severity,formality] env_setup.model_id=ihs_3comps