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
    "-meas",
    "--meas",
    default="erp",
    type=str,
    help="Measure to process",
)

args = parser.parse_args()

def grand_ave_erp(task, conditions) :
    
    evoked_one_condition = []

    for cond in conditions : 
        for subj in SUBJ_CLEAN :

            print("processing -->", subj)
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            
            # Drop EEg channels and equalize event number
            evoked_one_condition.append(epochs[cond].average()) 
            
        evokeds_all_subj = mne.combine_evoked(evoked_one_condition, 'equal')
        evokeds_all_subj.save(os.path.join(RESULT_PATH, 'meg/reports/',
                                           'sub-all/erp', 
                                           'sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.fif'.format(task, cond), 
                                           overwrite=True))

def grande_ave_tfr(task, conditions): 

    # Compute freqs from 2 - 60 Hz
    freqs = np.logspace(*np.log10([2, 120]))
    n_cycles = freqs / 2.  # different number of cycle per frequency

    all_tfr_power = []

    for cond in conditions :

        for subj in SUBJ_CLEAN : 

            _, epo_path = get_bids_file(RESULT_PATH, stage='AR_epo', task=task, subj=subj)
            epochs = mne.read_epochs(epo_path)
            epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')
            
            power = tfr_morlet(epochs[cond], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                    return_itc=False, decim=3, n_jobs=None, average=True)
            
            all_tfr_power.append(power)
            
        print(len(all_tfr_power))

        grand_average_tfr = mne.grand_average(all_tfr_power)
        grand_average_tfr.save(os.path.join(RESULT_PATH, 'meg/reports', 
                                            'sub-all/psd', 
                                            'sub-all_task-{}_run-all_cond-{}_meas-grandave-tfr.h5'.format(task, cond)))

if __name__ == "__main__" :

    task = args.task
    meas = args.meas

    if task == 'LaughterActive':
        conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']

    elif task == 'LaughterPassive':
        conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'EnvPosed',
                      'ScraReal', 'ScraPosed']

    if meas == 'erp' :
        grand_ave_erp(task, conditions)
    elif meas == 'tfr' :
        grande_ave_tfr(task, conditions)

