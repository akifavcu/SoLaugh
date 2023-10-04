import mne 
import pickle
import argparse
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, FREQS_LIST, PREPROC_PATH, FREQS_NAMES
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
parser.add_argument(
    "-tmin",
    "--tmin",
    default=0.0,
    type=float,
    help="Tmin to compute",
)
parser.add_argument(
    "-tmax",
    "--tmax",
    default=0.5,
    type=float,
    help="Tmin to compute",
)
args = parser.parse_args()

def plot_sf(sub_name, run_name, stage, conditions) :
    ######## PLOT TOPOMAPS ########

    fig_save = FIG_PATH + 'ml/results_single_feat/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq_{}{}.jpeg'.format(sub_name, task, 
                                                                                                                run_name, conditions,
                                                                                                                'scores', freq_name, stage)

    
    # Load one epoch for channel info
    _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj='01', stage="AR_epo")
    epochs = mne.read_epochs(path_epochs, verbose=None)

    # initialize figure
    fig, ax_topo = plt.subplots(1, len(FREQS_NAMES), figsize=(30, 30))

    for i, freq_name in enumerate(FREQS_NAMES): 
        
        # Load data
        save_scores = RESULT_PATH + 'meg/reports/sub-all/ml/results_single_feat/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq_{}{}.pkl'.format(sub_name, task, 
                                                                                                                                run_name, conditions,
                                                                                                                                'scores', freq_name, stage)
        with open(save_scores, 'rb') as f:
            all_scores = pickle.load(f)

        img, _ = mne.viz.plot_topomap(all_scores, epochs.info, axes=ax_topo[i], show=False, 
                                cmap='bwr', extrapolate='head',
                                sphere=(0, 0.0, 0, 0.19), vlim = (0, 1),)
        
        ax_topo[i].set_title(freq_name)    
        
        # TODO : put the colorbar on the right
        cbar = plt.colorbar(
            ax=ax_topo[i], shrink=0.2, orientation='vertical', mappable=img,)

        cbar.set_label('Accuracy')

    plt.savefig(fig_save)
    plt.show(fig_save)

if __name__ == "__main__" :

    task = args.task
    subj_list = SUBJ_CLEAN

    if task == 'LaughterActive':
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    # Select what conditions to compute (str)
    cond1 = args.cond1
    cond2 = args.cond2
    freq_name = args.freq
    conditions = cond1 + '-' + cond2
    condition_list = [cond1, cond2]

    tmin = args.tmin
    tmax = args.tmax

    frequency_list = []
    for i, freq in enumerate(FREQS_NAMES) : 
        if freq == freq_name :
            frequency_list.append(FREQS_LIST[i])

    sub_name = 'all'
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = 'ml-' + tmin_name + '-' + tmax_name
    conditions = cond1 + '-' + cond2

    plot_sf(sub_name, run_name, stage, conditions)
