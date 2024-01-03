#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughTimePerm
#SBATCH --mem=100G

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/erp_time_ttest-perm.py --task=LaughterActive --condition1=LaughPosed --condition2=BaselineZero
