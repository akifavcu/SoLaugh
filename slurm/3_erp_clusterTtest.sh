#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughClusterERP
#SBATCH --mem=64G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/erp_cluster_ttest.py --task=LaughterActive --condition1=Miss --baseline=True