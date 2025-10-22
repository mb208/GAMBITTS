#!/bin/bash
#SBATCH --account=tewaria1
#SBATCH --partition=spgpu
#SBATCH --job-name=ihs_dim
#SBATCH --output="gllogs/ihs_dim.out" 
#SBATCH --error="gllogs/ihs_dim.err" 
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-01:00:00

source /home/${USER}/.bashrc
conda activate nats 


python3 -m scripts.ihs_message_dimensions ihs.data_path='data/IHS/raw/2022/intervention/' \
 ihs.filename='2022 IHS Messages FOR ANALYSIS.csv' ihs.year=2022  \
 ihs.imputed_data='data/IHS/imputed/2022/2022_23_ihs_imputed.csv'