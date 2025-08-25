#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --output=prints/outputs/train-%j_%A_%a.out
#SBATCH --array=0-1
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

# Create a timestamp for the output file name.
timestamp=$(date +%s)

# Run the Python script with the SLURM_ARRAY_TASK_ID as the combo argument.  "$SLURM_ARRAY_TASK_ID"  ${SLURM_ARRAY_TASK_ID}
python3 $HOME/TropicalCNN/experiment.py --job_file jobs_to_do_1128_19_Mar_2025 --job_id "$SLURM_ARRAY_TASK_ID" > $HOME/TropicalCNN/prints/run_prints/training_check_lenets_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1
#python3 $HOME/TropicalCNN/experiment.py --job_file jobs_to_do_0636_17_Mar_2025 --job_id 0 > $HOME/TropicalCNN/prints/run_prints/zzz_attack_cifar100_effnet_0_${timestamp}.txt 2>&1
