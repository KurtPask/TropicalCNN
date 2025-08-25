#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=prints/outputs/train-%j_%A_%a.out
#SBATCH --array=0-11
#SBATCH --error=prints/errors/train-%j_%A_%a.err
#SBATCH --time=96:00:00 
#SBATCH --partition=dsag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G                           
#SBATCH --mail-type=END,FAIL                
#SBATCH --mail-user=kurt.pasque@nps.edu      

. /etc/profile
module load lang/python/3.13.0

source env_18Feb25/bin/activate

timestamp=$(date +%s)

python3 $HOME/TropicalCNN/experiment.py --combo "$SLURM_ARRAY_TASK_ID" --task evaluate --at yes --batch_size 50 > $HOME/TropicalCNN/prints/run_prints/evaluate_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1