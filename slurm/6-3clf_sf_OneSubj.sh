#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughSingleFeat
#SBATCH --mem=200G
#SBATCH --cpus-per-task=5

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/models/classification_sf_one_subj.py --task=LaughterPassive --tmin=0.0 --tmax=0.2   --freq=$1
