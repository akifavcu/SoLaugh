#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughPreproc
#SBATCH --mem=32G

module load python/3.10

$HOME/laughter_meg/bin/mne_bids_pipeline $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/data/preproc_config.py --subject=$1
