#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=GrandAveERP
#SBATCH --mem=200G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/grand_average_erp.py --task=LaughterActive --condition=LaughPosed
