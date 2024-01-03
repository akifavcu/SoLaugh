#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughSingleFeat
#SBATCH --mem=500G
#SBATCH --cpus-per-task=15
#SBATCH --output=clf_sf-test.out
#SBATCH --error=clf_sf-test.err

module load python/3.10

$HOME/laughter_meg/bin/python $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/models/classification_single_feature.py --task=LaughterPassive --tmin=0.7 --tmax=0.9 --freq=alpha --compute_hilbert=False
