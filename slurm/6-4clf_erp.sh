#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=3:00:00
#SBATCH --job-name=SoLaughSF-ERP
#SBATCH --mem=200G
#SBATCH --cpus-per-task=5

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/models/classification_sf_erp.py --tmin=0.7 --tmax=0.75 --compute_erp=False
