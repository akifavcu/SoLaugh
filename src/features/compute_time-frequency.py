# Importation
import mne
import os
import scipy.stats
import pickle
import matplotlib
import numpy as np
import argparse
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, RESULT_PATH, EVENTS_ID, FIG_PATH, FREQS_LIST, FREQS_NAMES
from matplotlib.backends.backend_pdf import PdfPages 

import numpy as np
from matplotlib import pyplot as plt

from mne.baseline import rescale
from mne.time_frequency import (
    tfr_multitaper,
    tfr_stockwell,
    tfr_morlet,
    tfr_array_morlet,
    AverageTFR,
)
from mne.viz import centers_to_edges


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
    "--cond",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-meas",
    "--meas",
    default="zscore",
    type=str,
    help="Mode to compute TF",
)
args = parser.parse_args()


def main(task, cond, meas, save=True) :    
    ROIS = ['MLO', 'MRO',
        'MLP', 'MRP',
        'MLT', 'MRT',
        'MLC', 'MRC',
        'MLF', 'MRF',
        'MZ']

    freqs = np.arange(2.0, 40.0, 3.0)
    vmin, vmax = -0.6, 0.6 

    filename = 'tf/sub-all_task-{}_run-all_cond-{}_meas-time-frequency_map-{}.pdf'.format(task, cond, meas)
    pdf = PdfPages(FIG_PATH + filename)

    for roi in ROIS : 
        
        all_epochs = []

        for subj in SUBJ_CLEAN :
            
            chan_selected = []

            print("processing -->", subj)

            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            
            if subj == SUBJ_CLEAN[0]:
                head_info = epochs.info['dev_head_t']
            else:
                epochs.info['dev_head_t'] = head_info
                
            all_epochs.append(epochs[cond])
            
        epochs_all_subj = mne.concatenate_epochs(all_epochs) # TODO SAVE THIS TO USE LOCAL
        
        for chan in epochs.info['ch_names'] :
            if roi in chan : 
                chan_selected.append(chan) # Find channel of interest        
        
        n_cycles = freqs/2
        
        print(len(chan_selected))
        print(epochs_all_subj.get_data().shape)
    
        ######## COMPUTE TF ON EPOCHS
        ######## 1. compute morlet on every channel separately
        power = tfr_morlet(
        epochs_all_subj, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=False, picks=chan_selected
        )
        
        fig, ax = plt.subplots(len(chan_selected), figsize=(10,70))
        ######## 2. Average Trials
        print('1st power', power.data.shape)
        avgpower = power.average()
        print('avg power ', avgpower.data.shape)
        
        # Plot average trial for every channel
        avgpower.plot(baseline=(-0.5, 0.0), mode=meas, axes=ax)
        plt.suptitle('All channels ROI ' + roi)
        
        if save == True : 
            plt.savefig(pdf, format='pdf') 
        plt.show()
        
        ######## 3. Average Channel
        avgpower_chan = np.mean(avgpower.data, axis=0)
        avgpower_chan = np.reshape(avgpower_chan, (1, avgpower_chan.shape[0], avgpower_chan.shape[1]))
        print('avg power channel ', avgpower_chan.shape)
        
        # Plot Average channel
        title_fig2 = "TFR calculated on a numpy array ROI " + roi
        # Baseline the output
        rescale(avgpower_chan, epochs.times, (-0.5, 0.0), mode=meas, copy=False)
        fig, ax = plt.subplots()
        x, y = centers_to_edges(epochs.times * 1000, freqs)
        mesh = ax.pcolormesh(x, y, avgpower_chan[0], cmap="RdBu_r") #, vmin=vmin, vmax=vmax)
        ax.set_title(title_fig2)
        ax.set(ylim=freqs[[0, -1]], xlabel="Time (ms)")
        fig.colorbar(mesh)
        plt.tight_layout()
        
        if save == True : 
            plt.savefig(pdf, format='pdf') 
        plt.show()

    pdf.close()

if __name__ == "__main__" :

    task = args.task 
    cond = args.cond
    meas = args.meas
    main(task, cond, meas, save=True)
