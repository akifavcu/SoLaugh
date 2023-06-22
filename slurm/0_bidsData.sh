#!/bin/bash
#SBATCH --job-name=SoLaughBIDS
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00     
#SBATCH --mem=20G  
#SBATCH --output=S-subtrain-%j.out
#SBATCH --error=S-subtrain-%j.err

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/data/bids_data.py