import mne 
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN
from mne.time_frequency import (tfr_morlet, AverageTFR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)

args = parser.parse_args()

def compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, event_id, task, FREQS_LIST) :
    
    power_list = []
    FREQS = [x for x in range(len(FREQS_LIST))]

    for freqs in FREQS : 
        for l_freq, h_freq in FREQS_LIST :
            print('Processing freqs -->', l_freq, h_freq)
            
            list_evo_subj = []

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
                
                list_evo_run = []
                
                for run in RUN_LIST :
                    print("=> Process run :", run)
                    
                    # Take raw data filtered : i.e. NO ICA 
                    _, raw_path = get_bids_file(PREPROC_PATH, stage='proc-filt_raw', task=task, run=run, subj=subj)
                    raw = mne.io.read_raw_fif(raw_path, preload=True)
                    raw = ica.apply(raw)

                    epochs_psds = []

                    freq_name = FREQS_NAMES[FREQS]

                    info = raw.info
                    raw_filter = raw.copy()
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
                                    
                    # Save epochs
                    stage = 'psd'
                    extension = '.fif'
                    # TODO : All epochs files should end with -epo.fif, -epo.fif.gz, _epo.fif or _epo.fif.gz
                    _, psd_path = get_bids_file(RESULT_PATH, subj, stage, task, measure=freq_name)
                    epochs_hilb.save(psd_path, overwrite=True)
                    
                    # Create Evokeds per run
                    list_evo_run.append(epochs_hilb['LaughReal'].average())
                    
                    # TODO: Drop bad epochs
                    tfr_data = epochs_hilb.get_data()
                    tfr_data = tfr_data * tfr_data.conj()  # compute power
                    tfr_data = np.mean(tfr_data, axis=0)  # average over epochs
                    
                    # TODO : save epochs_psds in pickle file
                    epochs_psds.append(epochs_hilb.get_data())
                
                # Combine evokeds per subj
                evoked_run = mne.combine_evoked(list_evo_run, weights='nave')
                list_evo_subj.append(evoked_run)
                
                epochs_psds = np.array(epochs_psds)
                epochs_psds = np.mean(epochs_psds, axis=3).transpose(1, 2, 0)
                print(epochs_psds.shape)
                
                # Compute power across all freq
                #power = AverageTFR(epochs_hilb.info, epochs_psds, epochs_hilb.times, FREQS_LIST, nave=len(epochs_hilb))  
                #power_list.append(power) # do it for each subj and each run

            # Average evoked across subj
            evoked_subj = mne.combine_evoked(list_evo_subj, weights='equal')

    return epochs_hilb

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    
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
        
    epochs_psd = compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, event_id=event_id, task=task, FREQS_LIST=FREQS_LIST)
