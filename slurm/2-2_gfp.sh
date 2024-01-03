#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughGFP
#SBATCH --mem=64G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/gfp.py --task=LaughterPassive --condition1=ScraReal --condition2=ScraPosed
