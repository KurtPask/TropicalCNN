#!/bin/bash
#SBATCH --job-name=csvs
#SBATCH --output=prints/outputs/avg_epoch_time.out
#SBATCH --error=prints/errors/avg_epoch_time.err
#SBATCH --time=96:00:00 
#SBATCH --partition=primary
#SBATCH --mem=20G                           
#SBATCH --mail-type=END,FAIL                
#SBATCH --mail-user=kurt.pasque@nps.edu      

cd $HOME/TropicalCNN
. /etc/profile
module load lang/python/3.13.0
source env_18Feb25/bin/activate


python3 avg_epoch_time.py