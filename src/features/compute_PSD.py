import mne 
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, FIG_PATH, RUN_LIST
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

def compute_hilbert_psd(raw, subj, event_id, freq, freq_name, picks, task, condition) :
    # Hilbert transform
    # TODO: loop for each freq
    epochs_psds = []

    for l_freq, h_freq in FREQS_LIST :

        raw_filter = raw.copy()
        raw_filter.filter(l_freq, h_freq)
        raw_hilbert = raw_filter.apply_hilbert(envelope=True)
        
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)

        # Segmentation
        events = mne.find_events(raw)
        epochs = mne.Epochs(
            raw_hilbert,
            events=events,
            event_id=event_id,
            tmin=-0.5,
            tmax=1.5,
            baseline=(None, 0),
            picks=picks)

        epochs_psds.append(epochs.get_data())

        # Save psd file fior each freq
        _, save_psd = get_bids_file(RESULT_PATH, subj = subj, task=task, stage="psd-hilbert", condition=condition, measure=freq_name)
        
        with open(save_psd, "wb") as f:
            pickle.dump(epochs, f)
   
    epochs_psds = np.array(epochs_psds)
    epochs_psds = np.mean(epochs_psds, axis=3).transpose(1, 2, 0)

    print(epochs_psds.shape)

    return epochs_psds

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    condition = args.condition
    freq_name = args.frequency

    condition_list = [condition]
    event_id = dict()

    # TODO : Take all event_id depending on task
    for ev in EVENTS_ID :
        for conds in condition_list :
            if conds not in EVENTS_ID :
                raise Exception("Condition is not an event")
            if conds == ev :
                event_id[conds] = EVENTS_ID[ev]

    for idx, f_name in enumerate(FREQS_NAMES) :
        if  freq_name in f_name :
            freq = FREQS_LIST[idx]
            freq = list(freq)

    for subj in SUBJ_CLEAN :
        print("=> Process task :", task, "condition :", condition, "frequency :", freq)
        for run in RUN_LIST :
            # Take raw data filtered : i.e. NO ICA 
            _, raw_path = get_bids_file(PREPROC_PATH, stage='proc-filt_raw', task=task, run=run, subj=subj)
            raw = mne.io.read_raw_fif(raw_path, preload=True)
        
            epochs_psd = compute_hilbert_psd(raw, subj, event_id, freq, freq_name, task, condition)
