#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughEpochs
#SBATCH --mem=100G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/data/create_epochs_clean.py --subject=$1
