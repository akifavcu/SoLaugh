#!/bin/bash
#SBATCH --job-name=SoLaughArchive
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00     
#SBATCH --mem=16  
#SBATCH --output=S-subtrain-%j.out
#SBATCH --error=S-subtrain-%j.err

tar --create --xz /home/claraelk/nearline/def-kjerbi/claraelk/laughter_data/bids_data.tar.xz /home/claraelk/scratch/laughter_data/bids_data/