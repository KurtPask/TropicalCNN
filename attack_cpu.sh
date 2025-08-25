#!/bin/bash
#SBATCH --job-name=att_cpu
#SBATCH --output=prints/outputs/attack_cpu-%j_%A_%a.out
#SBATCH --array=0-99
#SBATCH --error=prints/errors/attack_cpu-%j_%A_%a.err
#SBATCH --time=96:00:00 
#SBATCH --partition=primary
#SBATCH --mem=20G                           
#SBATCH --mail-type=END,FAIL                
#SBATCH --mail-user=kurt.pasque@nps.edu      

cd $HOME/TropicalCNN
. /etc/profile
module load lang/python/3.13.0
source env_18Feb25/bin/activate
timestamp=$(date +%s)

python3 attack.py --combo 19 --att_index "$SLURM_ARRAY_TASK_ID" --att_total 100 > prints/run_prints/attack_cpu_19_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1

#python3 attack.py --combo 4 --att_index 0 --att_total 100 > prints/run_prints/test_${timestamp}.txt 2>&1
