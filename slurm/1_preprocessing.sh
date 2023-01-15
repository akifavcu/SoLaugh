#SBATCH --account=def-kjerbi
#SBATCH --time=12:00:00
#SBATCH --job-name=SoLaughPreproc
#SBATCH --mem=32G

module load python/3.10.0

$HOME/laughter_meg/bin/mne-bids-pipeline $HOME/projects/def-kjerbi/claraelk/SoLaugh/src/config.py