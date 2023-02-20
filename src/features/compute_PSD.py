import mne 
import os
import pickle
import argparse
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH

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
    help="First condition",
)

parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Second condition",
)

parser.add_argument(
    "-freq",
    "--frequency",
    default="alpha",
    type=str,
    help="Frequency to compute",
)

args = parser.parse_args()

def compute_hilbert_PSD(epochs, event_id, freq, freq_name, picks, task) :
    # Hilbert transform
    # Compute epoch psd for each frequency
    # Save each epoch psd file in pickle format for each frequency
 
    l_freq = freq[0]
    h_freq = freq[1]

    epochs_filter = epochs.copy()
    epochs_filter.filter(l_freq, h_freq, picks = picks)
    epochs_hilbert = epochs_filter.apply_hilbert(picks = picks)
    
    # Separate epoch per conditions
    # We concatenate psd data for each freq
    epochs_psd.append(epochs_hilbert.get_data())

    epochs_psd = np.array(epochs_psd)
    epochs_psd = np.mean(epochs_psd, axis=3).transpose(1, 2, 0)
    print(epochs_psd.shape)

    # Save
    _, save_psd = get_bids_file(RESULT_PATH, task=task, stage="psd", condition=conditions, measure=freq_name)
    
    with open(save_psd, "wb") as f:
        pickle.dump(epochs_psd, f)

    return epochs_psd

#def compute_morlet_psd :
    # Morlet wavelet
    # Cycles
    # power = tfr_morlet(epochs, freqs=freqs,
    #                   n_cycles=n_cycles, return_itc=False)
    
if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    cond1 = args.condition1
    cond2 = args.condition2
    freq_name = args.frequency

    conditions = cond1 + '-' + cond2
    condition_list = [cond1, cond2]
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

    print("=> Process task :", task, "conditions :", cond1, "&", cond2, "frequency :", freq)

    # Import ERP files path
    _, save_epoch_concat = get_bids_file(RESULT_PATH, task=task, stage="erp-concat", condition=conditions)

    with open(save_epoch_concat, "rb") as f:
        epochs_concat = pickle.load(f)

    # Need to check if ave is = to this process
    compute_hilbert_PSD(epochs_concat, event_id, freq, freq_name, picks, task)
