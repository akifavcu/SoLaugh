import mne 
import os
import argparse
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from src.utils import get_bids_file, compute_ch_adjacency
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, FIG_PATH, ACTIVE_RUN, PASSIVE_RUN
from mne.time_frequency import (tfr_morlet, AverageTFR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-cond",
    "--condition",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-subj",
    "--subject",
    default="01",
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

def grande_ave_tfr(subj, task, cond): 
    # Compute freqs from 2 - 60 Hz
    freqs = np.logspace(*np.log10([2, 60]))
    n_cycles = freqs / 2.  # different number of cycle per frequency

    all_tfr_power = []

    for subj in SUBJ_CLEAN : 
        _, epo_path = get_bids_file(RESULT_PATH, stage='AR_epo', task=task, subj=subj)
        epochs = mne.read_epochs(epo_path)
        epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')
        
        power = tfr_morlet(epochs[cond], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=3, n_jobs=None, average=True)
        
        all_tfr_power.append(power)
        
    print(len(all_tfr_power))

    grand_average_tfr = mne.grand_average(all_tfr_power)
    grand_average_tfr.save('/home/claraelk/scratch/laughter_data/results/meg/reports/sub-all/psd/sub-all_task-{}_run-all_cond-{}_meas-test_grandave-tfr.h5'.format(task, cond))

    vmin, vmax = -3.0, 3.0  # Define our color limits.

    grand_average_tfr.plot(
        [0],
        baseline=(0.0, 0.1), 
        mode="mean",
        vmin=vmin,
        vmax=vmax,
        title="Using Morlet wavelets and EpochsTFR",
        show=False,
    )
    plt.savefig('/home/claraelk/scratch/laughter_data/results/meg/reports/figures/psd/sub-all_task-{}_run-all_cond-{}_meas-test_grandave-tfr.png'.format(task, cond))

    return grand_average_tfr

if __name__ == "__main__" :

    cond = args.condition
    task = args.task
    subj = args.subject

    grand_average_tfr = grande_ave_tfr(subj, task, cond)

