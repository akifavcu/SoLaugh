#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughfooof
#SBATCH --mem=100G
#SBATCH --output=fooof.out
#SBATCH --error=fooof.err


module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/compute_fooof.py --task=LaughterPassive