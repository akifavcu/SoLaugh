#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughSTC
#SBATCH --mem=500G
#SBATCH --cpus-per-task=10



module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/features/source_clustering.py --compute_cluster=True