import mne 
from src.utils import get_bids_file
from src.params import PREPROC_PATH

def ave_epochs(PREPROC_PATH, subj_list, run_list, cond1, cond2, stage) :
    
    epochs_concat = []

    for subj in subj_list :
        for run in run_list :
            epochs_file = get_bids_file(PREPROC_PATH, subj, run, stage)
            epochs = mne.read_epochs(epochs_file)
            print(epochs.info)

            # Average each condition
            epochs_cond1 = epochs[cond1].average()
            epochs_cond2 = epochs[cond2].average()

            epochs_concat = mne.concatenate_epochs([epochs]) # TODO : See if problem with head location

    return epochs_cond1, epochs_cond2, epochs_concat

#def compute_PSD(epochs_cond1, epochs_cond2, epochs_concat) :
    # See https://natmeg.se/mne_timefrequency/MNE_timefrequency.html
    # Compute epoch psd for each frequency
    # Save each epoch psd file in h5 format for each frequency

    # For each frequency :
    #frequencies = np.arange(5., 120., 2.5)  # define frequencies of interest
    #n_cycles = frequencies / 5.  # different number of cycle per frequency
    #Fs = raw.info['sfreq']  # sampling in Hz
    #decim = 5

    #data = epochs_stim.get_data()
    #from mne.time_frequency import induced_power
    #power, phase_lock = induced_power(data, Fs=Fs, frequencies=frequencies,
    #                              n_cycles=n_cycles, n_jobs=4, use_fft=False,
    #                              decim=decim, zero_mean=True)
    #power.shape, data.shape, frequencies.shape
    # Save 

def plot_PSD(epochs_cond1, epochs_cond2, epochs_concat) :
    
    # Plot PSD for condition 1
    fig_cond1_psd = epochs_cond1.plot_psd_topomap(ch_type='mag', normalize=False) 

    # Plot PSD for condition 2
    fig_cond2_psd = epochs_cond2.plot_psd_topomap(ch_type='mag', normalize=False)

    # Plot PSD for both conditions
    fig_concat_psd = epochs_concat.plot_psd_topomap(ch_type='mag', normalize=False)

    return fig_cond1_psd, fig_cond2_psd, fig_concat_psd

if __name__ == "__main__" :

    # Select subjects and runs and stage
    subj_list = ["01", "02"]
    run_list = ["07"]
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = "LaughReal"
    cond2 = "LaughPosed"
    picks = "meg" # Select MEG channels
    event_id = {'LaughReal' : 11, 'LaughPosed' : 12}

# Need to check if ave is = to this process
epochs_cond1, epochs_cond2, epochs_concat = ave_epochs(PREPROC_PATH, subj_list, run_list, cond1, cond2, "epo")

fig_cond1_psd, fig_cond2_psd, fig_concat_psd = plot_PSD(epochs_cond1, epochs_cond2, epochs_concat)
