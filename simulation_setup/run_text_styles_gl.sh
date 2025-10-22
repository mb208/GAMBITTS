#!/bin/sh

# script_name=$1

script_name=$1
style=$2
currdate=$(date +'%Y%m%d')
# seed=$3

sbatch --account=tewaria1 --partition=spgpu --mem=32g  --time=120:00:00 --mail-type=ALL --mail-user=marcbr@umich.edu --nodes=1  \
       --cpus-per-task=4 --gpus-per-node=1 \
       --job-name=$style --output="gllogs/$style_$currdate.out" --error="gllogs/$style_$currdate.err" $script_name $style
