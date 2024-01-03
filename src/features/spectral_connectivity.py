import mne 
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from mne_connectivity import spectral_connectivity_epochs

from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, PREPROC_PATH, FREQS_LIST, FREQS_NAMES
from src.utils import get_bids_file

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-cond1",
    "--condition1",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-freq",
    "--freq",
    default="alpha",
    type=str,
    help="Frquency to compute",
)
args = parser.parse_args()

def upload_data(cond1, cond2, freq_name) : 
    print('Load Data')

    all_epochs = []
    stage = 'psd_epo'

    for subj in subj_list : 
        print('Subj -->', subj)

        for run in run_list : 
            _, psd_path = get_bids_file(RESULT_PATH, stage=stage, subj=subj, task=task, run=run, measure=freq_name)
            epochs_run = mne.read_epochs(psd_path)
            sfreq = epochs_run.info['sfreq'] 
            
            # Set manually head mouvement
            if subj == subj_list[0] and run == run_list[0]:
                head_info = epochs_run.info['dev_head_t']
            else:
                epochs_run.info['dev_head_t'] = head_info

            all_epochs.append(epochs_run[cond1])

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs, sfreq

def compute_connectivity(method, epochs, fmin, fmax, sfreq) :
    print('Plot')
    print(method)
    
    sub_name = 'all'
    run_name = 'all'
    conditions = cond1
    stage = 'con'

    connectivity_path = FIG_PATH + 'connectivity/sub-{}_task-{}_run-{}_cond-{}_meas-{}_freq_{}{}.png'.format(sub_name, task, 
                                                                                                                run_name, conditions,
                                                                                                                method, freq_name, stage)

    con_wpli = spectral_connectivity_epochs(
    epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, tmin=0, mt_adaptive=False, n_jobs=1)

    fig, axs = plt.subplots(1, figsize=(14, 5), sharey=True)

    axs.imshow(con_wpli.get_data('dense'), vmin=0, vmax=1)
    axs.set_title("wPLI")
    axs.set_xlabel("Sensor 2")

    plt.savefig(connectivity_path)

    return con_wpli
        
if __name__ == "__main__" :

    subj_list = SUBJ_CLEAN
    task = args.task

    # Condition 1 and 2
    cond1 = args.condition1
    cond2 = args.condition2
    
    # Frequence selected
    freq_name = args.freq   
    method = 'wpli'

    for i, freq in enumerate(FREQS_NAMES) : 
        if freq == freq_name :
            fmin, fmax = FREQS_LIST[i][0], FREQS_LIST[i][1]

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    epochs, sfreq = upload_data( cond1, cond2, freq_name)
    con_wpli = compute_connectivity(method, epochs, fmin, fmax, sfreq)