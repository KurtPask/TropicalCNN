#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=prints/outputs/train-%j_%A_%a.out
#SBATCH --array=12-17
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

# Run the Python script with the SLURM_ARRAY_TASK_ID as the combo argument.
#python3 $HOME/TropicalCNN/train.py --combo "$SLURM_ARRAY_TASK_ID" --lr 0.001 --num_epochs 50 --at True > $HOME/TropicalCNN/prints/run_prints/train_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1
#python3 $HOME/TropicalCNN/train.py --combo 0 --lr 0.001 --num_epochs 50 --at True > $HOME/TropicalCNN/prints/run_prints/train_0_${timestamp}.txt 2>&1
#python3 $HOME/TropicalCNN/experiment.py --combo 4 --task train --at yes --num_epochs 50 --batch_size 50 > $HOME/TropicalCNN/prints/run_prints/train_7_${timestamp}.txt 2>&1
python3 $HOME/TropicalCNN/experiment.py --combo "$SLURM_ARRAY_TASK_ID" --task train --at yes --num_epochs 50 --batch_size 25 > $HOME/TropicalCNN/prints/run_prints/train_adv_smaxout_${SLURM_ARRAY_TASK_ID}_${timestamp}.txt 2>&1
