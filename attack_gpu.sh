#!/bin/bash
#SBATCH --job-name=att_gpu
#SBATCH --output=prints/outputs/attack_gpu-%j_%A_%a.out
#   #SBATCH --array=9-11
#SBATCH --error=prints/errors/attack_gpu-%j_%A_%a.err
#SBATCH --time=96:00:00 
#SBATCH --partition=dsag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                           
#SBATCH --mail-type=END,FAIL                
#SBATCH --mail-user=kurt.pasque@nps.edu      

cd $HOME/TropicalCNN

. /etc/profile
module load lang/python/3.13.0

source env_18Feb25/bin/activate

timestamp=$(date +%s)
#   export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


#python3 experiment.py --combo "$SLURM_ARRAY_TASK_ID" --task attack --at yes --batch_size 25 > prints/run_prints/zz_attack_at_gpu_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1
python3 experiment.py --combo 4 --task attack --at yes --batch_size 20 > prints/run_prints/zzzzattack_gpu_4_${timestamp}.txt 2>&1
