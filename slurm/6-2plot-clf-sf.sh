#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=3:00:00
#SBATCH --job-name=SoLaughPlot
#SBATCH --mem=50G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/visualization/plot_single-feat.py --condition1=Good --tmin=0.0 --tmax=0.2