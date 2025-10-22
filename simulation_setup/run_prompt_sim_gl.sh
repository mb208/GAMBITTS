script_name=$1
style=$2

outpath="gllogs/prompt_sim_${style}.out"
errorpath="gllogs/prompt_sim_${style}.err" 
sbatch --account=tewaria1 --partition=spgpu --mem=8g  --time=00-12:00:00 --mail-type=ALL --mail-user=marcbr@umich.edu --nodes=1  \
       --cpus-per-task=4 --gpus-per-node=1 \
       --job-name=$style --output=$outpath --error=$errorpath $script_name $style
       
