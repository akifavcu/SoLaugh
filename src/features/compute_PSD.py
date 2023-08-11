import mne 
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH
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
    "-psd",
    "--psd",
    default="morlet",
    type=str,
    help="Time Frequency analysis",
)
args = parser.parse_args()

def compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, event_id, task, FREQS_LIST) :
    
    power_list = []
    FREQS = [x for x in range(len(FREQS_LIST))]
    idx = 0 #TODO : Change that

    for l_freq, h_freq in FREQS_LIST :
        print('Processing freqs -->', l_freq, h_freq)
        
        for subj in SUBJ_CLEAN:
            print("=> Process task :", task, 'subject', subj)

            sub_id = 'sub-' + subj
            subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id)

            if not os.path.isdir(subj_path):
                os.mkdir(subj_path)
                print('BEHAV folder created at : {}'.format(subj_path))
            else:
                print('{} already exists.'.format(subj_path))

            # Take ica data
            _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
            ica = mne.preprocessing.read_ica(ica_path)

            _, reject_path = get_bids_file(RESULT_PATH, stage="AR_epo", task=task, subj=subj, measure='log')
            with open(reject_path, 'rb') as f:
                reject = pickle.load(f)  
                        
            for run in RUN_LIST :
                print("=> Process run :", run)
                
                # Take raw data filtered : i.e. NO ICA 
                _, raw_path = get_bids_file(PREPROC_PATH, stage='proc-filt_raw', task=task, run=run, subj=subj)
                raw = mne.io.read_raw_fif(raw_path, preload=True)
                raw_filter = raw.copy()
                raw_filter = ica.apply(raw_filter)

                epochs_psds = []

                freq_name = FREQS_NAMES[idx]

                info = raw_filter.info
                raw_filter.filter(l_freq, h_freq)
                raw_hilbert = raw_filter.apply_hilbert(envelope=True)

                picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)
                
                # Segmentation
                events = mne.find_events(raw)
                epochs_hilb = mne.Epochs(
                    raw_hilbert,
                    events=events,
                    event_id=event_id,
                    tmin=-0.5,
                    tmax=1.5,
                    baseline=(None, 0),
                    picks=picks)    
                
                # Auto-reject epochs
                epochs_hilb.drop_bad(reject=reject)

                # Save epochs
                stage = 'psd_epo'
                _, psd_path = get_bids_file(RESULT_PATH, stage=stage, subj=subj, task=task, run=run, measure=freq_name)
                epochs_hilb.save(psd_path, overwrite=True)
                
                #tfr_data = epochs_hilb.get_data()
                #tfr_data = tfr_data * tfr_data.conj()  # compute power
                #tfr_data = np.mean(tfr_data, axis=0)  # average over epochs

                
        idx += 1

    return epochs_hilb

def compute_morlet_psd(SUBJ_CLEAN, task, event_id) :
    all_evoked = [] 
    
    for i, condition in enumerate(event_id.keys()):
        for subj in SUBJ_CLEAN:
            print("=> Process task :", task, 'subject', subj)

            _, epo_path = get_bids_file(PREPROC_PATH, stage='proc-clean_epo', task=task, subj=subj)
            epochs = mne.read_epochs(epo_path)
            epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')

            # Average for one condition
            all_evoked.append(epochs[condition].average())

        # Combine all subjects
        evokeds = mne.combine_evoked(all_evoked, weights='equal')

        # Compute freqs from 2 - 60 Hz
        freqs = np.logspace(*np.log10([2, 60]))
        n_cycles = freqs / 2.  # different number of cycle per frequency
        power = tfr_morlet(evokeds, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                return_itc=False, decim=3, n_jobs=None)
        
        # Save power 
        stage = 'psd'
        _, psd_path = get_bids_file(RESULT_PATH, 
                                    stage=stage, 
                                    subj=subj, 
                                    task=task, 
                                    condition = condition, 
                                    measure='morlet')
        power.save(psd_path, overwrite=True)

        # Plot topomaps
        fig_path = FIG_PATH + 'psd/subj-all_task-{}_cond-{}_meas-morletPSD_topomap'.format(task, condition)
        title_fig = 'Topomaps Morlet wavelet - condition {} - task {}'.format(condition, task)
        fig, axes = plt.subplots(1, 5, figsize=(30, 10))
        topomap_kw = dict(ch_type='mag', tmin=0.5, tmax=1.5, baseline=(-0.5, 0),
                        mode='logratio', show=False)
        
        plot_dict = dict(Delta=dict(fmin=2, fmax=4), Theta=dict(fmin=5, fmax=7),
                        Alpha=dict(fmin=8, fmax=12), Beta=dict(fmin=13, fmax=25),
                        Gamma=dict(fmin=26, fmax=60))
        
        for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
            power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
            ax.set_title(title)

        plt.suptitle(title_fig, size = 30)
        fig.tight_layout()
        fig.savefig(fig_path)
        #fig.show()     
    
    return power

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    psd = args.psd

    if task == 'LaughterActive' :
        RUN_LIST = ACTIVE_RUN
    elif task == 'LaughterPassive':
        RUN_LIST = PASSIVE_RUN

    for ev in EVENTS_ID :
        if task == 'LaughterActive' :
            event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66}
        elif task == 'LaughterPassive' : 
            event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 'ScraReal' : 31, 
                        'EnvPosed' : 22, 'ScraPosed' : 32,}
        
    if psd == 'hilbert' : 
        epochs_psd = compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, event_id=event_id, task=task, FREQS_LIST=FREQS_LIST)
    
    elif psd == 'morlet' : 
        power = compute_morlet_psd(SUBJ_CLEAN, task=task, event_id=event_id)
