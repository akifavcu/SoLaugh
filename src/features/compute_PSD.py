import mne 
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, FIG_PATH
from mne.time_frequency import tfr_morlet, AverageTFR

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
    "-freq",
    "--frequency",
    default="alpha",
    type=str,
    help="Frequency to compute",
)

args = parser.parse_args()

def compute_hilbert_psd(epochs, subj, event_id, freq, freq_name, picks, task, condition) :
    # Hilbert transform
    # Separate epoch per conditions
    l_freq = freq[0]
    h_freq = freq[1]

    epochs_filter = epochs.copy()
    epochs_filter.pick_types(meg=True, ref_meg=False)
    epochs_filter.filter(l_freq, h_freq, picks = picks)
    epochs_hilbert = epochs_filter.apply_hilbert(picks = picks, envelope=True)
    
    tfr_data = epochs_hilbert.get_data()
    tfr_data = tfr_data * tfr_data.conj()  # compute power
    tfr_data = np.mean(tfr_data, axis=0)  # average over epochs
    data[:, idx] = tfr_data
    power = AverageTFR(info, data, epochs.times, freq, nave=n_epochs)

    # Save
    _, save_psd = get_bids_file(RESULT_PATH, subj = subj, task=task, stage="psd-hilbert", condition=condition, measure=freq_name)
    
    with open(save_psd, "wb") as f:
        pickle.dump(epochs_psd, f)

    return tfr_data

def compute_morlet_psd(epochs, event_id, freq, freq_name, picks, task, condition) :
    
    epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')

    # Morlet wavelet
    n_cycles = freq / 2.
    power = tfr_morlet(epochs, freqs=freq,
                    n_cycles=n_cycles, return_itc=False, average=True, picks=picks)
    print(type(power))
    print(power)

    _, save_psd = get_bids_file(RESULT_PATH, task=task, stage="psd-morlet", condition=condition, measure=freq_name)
    
    with open(save_psd, "wb") as f:
        pickle.dump(power, f)

    power.plot([0], baseline=(0., 0.1), mode='mean', vmin=-3, vmax=3,
              title='Using Morlet wavelets and EpochsTFR', show=False)
              
    plt.savefig(FIG_PATH + 'subj-all_task-{}_cond-{}_meas-PSD-Morlet.png')
    plt.show()

    return power

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    condition = args.condition
    freq_name = args.frequency

    condition_list = [condition]
    event_id = dict()
    picks = "meg" # Select MEG channels

    for ev in EVENTS_ID :
        for conds in condition_list :
            if conds not in EVENTS_ID :
                raise Exception("Condition is not an event")
            if conds == ev :
                event_id[conds] = EVENTS_ID[ev]

    for idx, f_name in enumerate(FREQS_NAMES) :
        if  freq_name in f_name :
            freq = FREQS_LIST[idx]
            freq = np.asarray(freq)


    for subj in SUBJ_CLEAN :
        print("=> Process task :", task, "condition :", condition, "frequency :", freq)
        epochs_path = get_bids_file(PREPROC_PATH, stage='proc-clean_epo', task=task)
        epochs = mne.read_epochs(epochs_path)
        
        epochs_psd = compute_hilbert_psd(epochs, subj, event_id, freq, freq_name, picks, task, condition)
