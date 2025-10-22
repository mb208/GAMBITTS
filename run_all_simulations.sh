#!/bin/bash


############################################################################################################
######                         
######                            Runs all simulations for results found in 
######          Generator-Mediated Bandits: Thompson Sampling for GenAI-Powered Adaptive Interventions
######
############################################################################################################

# Main driver for all simulation runs
# Usage: ./run_all_simulations.sh

set -e  # Exit on first error
set -o pipefail
source /home/${USER}/.bashrc
conda activate nats

LOG_DIR="logs"
RESULTS_DIR="data/results"
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR


## Global Sim Parameters 
MC_ITERATIONS=250
HORIZON=1000
# Number of samples from generator used in poGAMBITT (varies in db_access simulations)
DB_SAMPLES=500
# Number of prompts in prompt_db (varies in scale_arms simulations)
n_prompts=5
# Default 2nd stage standard deviation level (varies in variance decomposition simulations)
sd_lvl=_sd_0
today=$(date +"%Y-%m-%d")


echo "Running simulations using parameters:"
echo "  MC_ITERATIONS     = $MC_ITERATIONS"
echo "  HORIZON           = $HORIZON"

# Simple 1-d simulations: outcome model depends on 1 dimension (correct and misspecified cases)
one_d_sim=false 
# Scaling arms in 3-d outcome linear environment
scale_arms=false
# Scaling 2nd stage variance in 3-d outcome linear environment
var_decomp_linear=false
# Scaling 2nd stage variance in 3-d outcome non-linear environment
var_decomp_nonlinear=false
# Simulations with database access (varying number of samples)
db_access_sim=false 
# Simulations to scale number of dimensions used 
scale_dimensions=false
# Simulations specific fogambitt agent (extended time horizons)
scale_fogambitt_arms=false
# Hedge models (1-d scenarios)
hedged_1d_sim=false 
# 3-D model with covariates
covariates=false


# --- Parse command-line args ---
for arg in "$@"; do
  case $arg in
    --all)
      one_d_sim=true
      scale_arms=true
      var_decomp_linear=true
      var_decomp_nonlinear=true
      db_access_sim=true
      scale_dimensions=true
      scale_fogambitt_arms=true
      hedged_1d_sim=true
      covariates=true
      ;;
    --one_d_sim) one_d_sim=true ;;
    --scale_arms) scale_arms=true ;;
    --var_decomp_linear) var_decomp_linear=true ;;
    --var_decomp_nonlinear) var_decomp_nonlinear=true ;;
    --db_access_sim) db_access_sim=true ;;
    --scale_dimensions) scale_dimensions=true ;;
    --scale_fogambitt_arms) scale_fogambitt_arms=true ;;
    --hedged_1d_sim) hedged_1d_sim=true ;;
    --covariates) covariates=true ;;
    --help|-h)
      echo "Usage: $0 [--all] [--one_d_sim] [--scale_arms] [--var_decomp_linear] [--var_decomp_nonlinear] [--db_access_sim] [--scale_dimensions] [--scale_fogambitt_arms] [--hedged_1d_sim] [--covariates]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

# --- Print what’s on (for debugging) ---
echo "one_d_sim=$one_d_sim"
echo "scale_arms=$scale_arms"
echo "var_decomp_linear=$var_decomp_linear"
echo "var_decomp_nonlinear=$var_decomp_nonlinear"
echo "db_access_sim=$db_access_sim"
echo "scale_dimensions=$scale_dimensions"
echo "scale_fogambitt_arms=$scale_fogambitt_arms"
echo "hedged_1d_sim=$hedged_1d_sim"
echo "covariates=$covariates"



if [[ "${one_d_sim}" == true ]]; then
    echo "Running 1-d simulations..."
    environment=ihs_linear_outcome_base$sd_lvl
    agents=ponats_linear_1comp,ensemble_sampling_1dim,fonats_linear_1comp

    sim_cfg=linear${sd_lvl}_sim_5_prompts
    sim_id=1d_experiment_linear${sd_lvl}_$today

    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environment"
    echo "Using agents: $agents"
    echo "log file: ./logs/${sim_id}_nats.txt"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                agent/covariates=severity,encouragement,formality,optimism,clarity agent.dim_batch=null  \
                simulation=$sim_cfg \
                environment=$environment \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                ~env_setup  \
                > ./logs/${sim_id}_gmb.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_gmb.txt"
    
    agents=one_stage_linear_cb,one_stage_mab
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environment \
                simulation.mc_iterations=$MC_ITERATIONS \
                simulation.horizon=$HORIZON  \
                simulation.sim_id=$sim_id \
                ~env_setup \
                > ./logs/${sim_id}_bl.txt 2>&1 &
    
    echo "Logs being written to ./logs/${sim_id}_bl.txt"
elif [[ "${scale_arms}" == true ]]; then
    environments=ihs_linear_outcome_3comps$sd_lvl
    agents=ensemble_sampling_3dim,ponats_linear_3comp,fonats_linear_3comp,one_stage_mab
    sim_configs=linear${sd_lvl}_sim_3_prompts,linear${sd_lvl}_sim_5_prompts,linear${sd_lvl}_sim_15_prompts,linear${sd_lvl}_sim_30_prompts,linear${sd_lvl}_sim_40_prompts
    sim_id=nprompts${sd_lvl}_linear_$today

    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    echo "Using agents: $sim_configs"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}.txt"

elif [[ "${var_decomp_linear}" == true ]]; then

############################################################################################################
######                           Scaling Variance on Generative Linear Environment
######
######
############################################################################################################
    __sd_lvl=_sd_0
    # Below are all combinations of agents and environments to be used in this experiment
    agents=ponats_linear_3comp,ensemble_sampling_3dim,fonats_linear_3comp,one_stage_mab
    sim_cfg=linear${__sd_lvl}_sim_5_prompts
    environments=ihs_linear_outcome_3comps${__sd_lvl}
    sim_id=2nd_stage_var_linear_$today
    echo "Using agents: $agents"
    echo "Using environments: $environments"
    echo "Using simulation config: $sim_cfg"
    echo "Running simulation  $sim_id with $__sd_lvl"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_1
    sim_cfg=linear${__sd_lvl}_sim_5_prompts
    environments=ihs_linear_outcome_3comps${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_2
    sim_cfg=linear${__sd_lvl}_sim_5_prompts
    environments=ihs_linear_outcome_3comps${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_3
    sim_cfg=linear${__sd_lvl}_sim_5_prompts
    environments=ihs_linear_outcome_3comps${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_4
    sim_cfg=linear${__sd_lvl}_sim_5_prompts
    environments=ihs_linear_outcome_3comps${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

elif [[ "${var_decomp_nonlinear}" == true ]]; then
############################################################################################################
######                Scaling Variance on Generative Neural Net Environment
######
######
############################################################################################################
    __sd_lvl=_sd_0
    # Below are all combinations of agents and environments to be used in this experiment
    agents=ponats_linear_3comp,ensemble_sampling_3dim,fonats_linear_3comp,one_stage_mab
    sim_cfg=nn${__sd_lvl}_sim_5_prompts
    environments=ihs_3comps_mlp_outcome_model${__sd_lvl}
    sim_id=2nd_stage_var_nn_$today
    echo "Using agents: $agents"
    echo "Using environments: $environments"
    echo "Using simulation config: $sim_cfg"
    echo "Running simulation  $sim_id with $__sd_lvl"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &
    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_1
    sim_cfg=nn${__sd_lvl}_sim_5_prompts
    environments=ihs_3comps_mlp_outcome_model${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &
    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_2
    sim_cfg=nn${__sd_lvl}_sim_5_prompts
    environments=ihs_3comps_mlp_outcome_model${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &
    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_3
    sim_cfg=nn${__sd_lvl}_sim_5_prompts
    environments=ihs_3comps_mlp_outcome_model${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${__sd_lvl}.txt"

    __sd_lvl=_sd_4
    sim_cfg=nn${__sd_lvl}_sim_5_prompts
    environments=ihs_3comps_mlp_outcome_model${__sd_lvl}
    echo "Running simulation  $sim_id with $__sd_lvl"
    echo "Using simulation config: $sim_cfg"
    echo "Using environments: $environments"
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                environment=$environments \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                simulation.store_dims=false ~env_setup \
                > logs/${sim_id}${__sd_lvl}.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}${sd_lvl}.txt"

elif [[ "${db_access_sim}" == true ]]; then
############################################################################################################
######                           Access to true conditional distribution of Z
######
######
############################################################################################################
    environments=ihs_linear_outcome_3comps$sd_lvl
    agents=ensemble_sampling_3dim,ponats_linear_3comp
    sim_cfg=linear${sd_lvl}_sim_${n_prompts}_prompts
    sim_id=db_access${sd_lvl}_linear_env_$today
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    # # # Varying N prompt_db samples for two stage agents
    python -m scripts.run_simulation --multirun \
                agent=$agents \
                simulation=$sim_cfg \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                environment=$environments \
                simulation.sim_id=$sim_id \
                simulation.n_samples=15,50,100,500,null \
                simulation.prompt_db=data/sim_params/prompt_db_20dim_950samples.csv.gzip \
                ~env_setup > ./logs/${sim_id}_gmb.txt 2>&1 &

    echo "Logs being written to ./logs/${sim_id}_gmb.txt"

    python -m scripts.run_simulation --multirun \
                agent=one_stage_mab \
                simulation=$sim_cfg \
                simulation.mc_iterations=$MC_ITERATIONS \
            simulation.horizon=$HORIZON \
                environment=$environments \
                simulation.sim_id=$sim_id \
                simulation.prompt_db=data/sim_params/prompt_db_20dim_950samples.csv.gzip \
                ~env_setup > ./logs/${sim_id}_mab.txt 2>&1 &

elif [[ "${scale_dimensions}" == true ]]; then

############################################################################################################
######
######      Comparing How Nats Varations Compare to One Stage Agents as Latnet Dimension Varies
######
######
############################################################################################################
    dims=3
    sim_id=scaling_dims${sd_lvl}_linear_env_$today
    environments=ihs_linear_outcome_${dims}dims$sd_lvl
    agents=ens_ponats_${dims}_dims,ponats_${dims}_dims,fonats_${dims}_dims,one_stage_mab
    sim_configs=linear_${dims}_dim${sd_lvl}_sim_${n_prompts}_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using simulation spec: $sim_configs"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}_${dims}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_${dims}.txt"
    dims=5
    environments=ihs_linear_outcome_${dims}dims$sd_lvl
    agents=ens_ponats_${dims}_dims,ponats_${dims}_dims,fonats_${dims}_dims,one_stage_mab
    
    sim_configs=linear_${dims}_dim${sd_lvl}_sim_${n_prompts}_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using simulation spec: $sim_configs"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}_${dims}.txt  2>&1 &
    echo "Logs being written to ./logs/${sim_id}_${dims}.txt"

    dims=10
    environments=ihs_linear_outcome_${dims}dims$sd_lvl
    agents=ens_ponats_${dims}_dims,ponats_${dims}_dims,fonats_${dims}_dims,one_stage_mab
    sim_configs=linear_${dims}_dim${sd_lvl}_sim_${n_prompts}_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using simulation spec: $sim_configs"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}_${dims}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_${dims}.txt"

    dims=15
    environments=ihs_linear_outcome_${dims}dims$sd_lvl
    agents=ens_ponats_${dims}_dims,ponats_${dims}_dims,fonats_${dims}_dims,one_stage_mab
    sim_configs=linear_${dims}_dim${sd_lvl}_sim_${n_prompts}_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using simulation spec: $sim_configs"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}_${dims}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_${dims}.txt"
    dims=full
    environments=ihs_linear_outcome_${dims}_dims$sd_lvl 
    agents=ens_ponats_${dims}_dims,ponats_${dims}_dims,fonats_${dims}_dims,one_stage_mab
    sim_configs=linear_${dims}_dim${sd_lvl}_sim_${n_prompts}_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using simulation spec: $sim_configs"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}_${dims}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_${dims}.txt"

elif [[ "${scale_fogambitt_arms}" == true ]]; then
############################################################################################################
######
######              Comparing How foGAMBITT Compares to One Stage Agents as Prompts Vary
######
######
############################################################################################################
    HORIZON_FGMB=10000
    __sd_lvl=_sd_0
    environments=ihs_linear_outcome_3comps$__sd_lvl
    agents=one_stage_linear_cb,one_stage_mab,fonats_linear_3comp
    sim_configs=linear${__sd_lvl}_sim_40_prompts
    sim_id=fogammbit_linear_env_40_$today
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS \
            simulation.horizon=$HORIZON_FGMB \
            simulation.sim_id=$sim_id \
            simulation.store_dims=false \
            ~env_setup \
                > ./logs/${sim_id}_${__sd_lvl}.txt  2>&1 &
    echo "Logs being written to ./logs/${sim_id}_${__sd_lvl}.txt"

    __sd_lvl=_sd_2
    environments=ihs_linear_outcome_3comps$__sd_lvl
    sim_configs=linear${__sd_lvl}_sim_40_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS \
            simulation.horizon=$HORIZON_FGMB \
            simulation.sim_id=$sim_id \
            simulation.store_dims=false \
            ~env_setup \
                > ./logs/${sim_id}_${__sd_lvl}.txt  2>&1 &
    echo "Logs being written to ./logs/${sim_id}_${__sd_lvl}.txt"


    __sd_lvl=_sd_4
    environments=ihs_linear_outcome_3comps$__sd_lvl
    sim_configs=linear${__sd_lvl}_sim_40_prompts
    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS \
            simulation.horizon=$HORIZON_FGMB \
            simulation.sim_id=$sim_id \
            simulation.store_dims=false \
            ~env_setup \
                > ./logs/${sim_id}_${__sd_lvl}.txt  2>&1 &
    echo "Logs being written to ./logs/${sim_id}_${__sd_lvl}.txt"

elif [[ "${hedged_1d_sim}" == true ]]; then

############################################################################################################
######                  Simple Scenario with misspecified outcome model using prompt flags 
######
######
############################################################################################################
    environment=ihs_linear_outcome_base$sd_lvl
    agents=ponats_linear_1comp_prompts,ensemble_sampling_1dim_prompts,fonats_linear_1comp_prompts

    sim_cfg=linear${sd_lvl}_sim_5_prompts
    sim_id=1d_experiment_linear_prompts${sd_lvl}_$today

    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environment"
    echo "Using agents: $agents"

    python -m scripts.run_simulation --multirun \
                agent=$agents \
                agent/covariates=severity,encouragement,formality,optimism,clarity \
                agent.dim_batch=null \
                simulation=$sim_cfg \
                environment=$environment \
                simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
                simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
                ~env_setup  \
                > ./logs/${sim_id}_nats.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}_nats.txt"
elif [[ "${covariates}" == true ]]; then
############################################################################################################
######                  3-D Scenario with covariates in outcome model
######
######
############################################################################################################
    environments=ihs_linear_outcome_3comps_ctx$sd_lvl
    agents=ensemble_sampling_3dim_ctx,ponats_linear_3comp_ctx,fonats_linear_3comp_ctx,one_stage_mab,one_stage_linear_cb
    sim_configs=linear${sd_lvl}_sim_5_prompts
    sim_id=covariates_exp${sd_lvl}_linear_$today

    echo "Running simulation with sim_id: $sim_id"
    echo "Using environments: $environments"
    echo "Using agents: $agents"
    echo "Using agents: $sim_configs"
    python -m scripts.run_simulation --multirun \
            agent=$agents \
            environment=$environments \
            simulation=$sim_configs \
            simulation.mc_iterations=$MC_ITERATIONS simulation.horizon=$HORIZON \
            simulation.sim_id=$sim_id simulation.n_samples=$DB_SAMPLES \
            simulation.store_dims=false ~env_setup \
                > ./logs/${sim_id}.txt  2>&1 &

    echo "Logs being written to ./logs/${sim_id}.txt"
fi