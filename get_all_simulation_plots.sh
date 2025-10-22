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

mkdir -p ./results/plots
mkdir -p ./results/plots/main_body
mkdir -p ./results/plots/supplement

# Run notebooks to generate simulation plots and save reults
jupyter nbconvert --to notebook --execute notebooks/plots_misspecification_1d.ipynb
jupyter nbconvert --to notebook --execute notebooks/plots_scaling_prompt_space.ipynb
jupyter nbconvert --to notebook --execute notebooks/plots_scaling_variance.ipynb
jupyter nbconvert --to notebook --execute notebooks/plots_varying_db_acess.ipynb
jupyter nbconvert --to notebook --execute notebooks/plots_scaling_dimesnions.ipynb
jupyter nbconvert --to notebook --execute notebooks/plots_fogambitt_simulation.ipynb

# Non experiment related plots
jupyter nbconvert --to notebook --execute plot_vae_score_correlations.ipynb

# The following requires gpu to run
# jupyter nbconvert --to notebook --execute plot_optimisim_embedding.ipynb

# Modify plots to match the paper style
jupyter nbconvert --to notebook --execute modify_mainbody_plots.ipynb
jupyter nbconvert --to notebook --execute modify_supplement_plots.ipynb