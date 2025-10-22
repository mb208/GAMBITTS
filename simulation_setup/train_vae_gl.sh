
script_name=$1
style=$2
data_folder=$3
filename=$4
currdate=$(date +'%Y%m%d')
# seed=$3
outpath="gllogs/vae_${style}.out"
errorpath="gllogs/vae_${style}.err" 
sbatch --account=tewaria1 --partition=spgpu --mem=16g  --time=120:00:00 --mail-type=ALL --mail-user=marcbr@umich.edu --nodes=1  \
       --cpus-per-task=4 --gpus-per-node=1 \
       --job-name=$style --output=$outpath --error=$errorpath $script_name $style $data_folder $filename
       