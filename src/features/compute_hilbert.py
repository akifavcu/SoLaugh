import mne 
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-tmin",
    "--tmin",
    default=-0.5,
    type=float,
    help="Minimum time",
)
parser.add_argument(
    "-tmax",
    "--tmax",
    default=1.5,
    type=float,
    help="Maximum time",
)
args = parser.parse_args()

def compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, task, FREQS_LIST, tmin, tmax) :
    
    FREQS = [x for x in range(len(FREQS_LIST))]
    idx = 0 #TODO : Change that

    for l_freq, h_freq in FREQS_LIST :
        print('Processing freqs -->', l_freq, h_freq)
        
        for i_subj, subj in enumerate(SUBJ_CLEAN):
            print("=> Process task :", task, 'subject', subj)

            sub_id = 'sub-' + subj
            subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id, 'hilbert')

            if not os.path.isdir(subj_path):
                os.mkdir(subj_path)
                print('BEHAV folder created at : {}'.format(subj_path))
            else:
                print('{} already exists.'.format(subj_path))

            # Take ica data
            _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
            ica = mne.preprocessing.read_ica(ica_path)

            # Select rejection threshold associated with each run
            # TODO : apply threshold per run
            print('Load reject threshold')
            _, reject_path = get_bids_file(RESULT_PATH, stage="AR_epo", task=task, subj=subj, measure='log')

            with open(reject_path, 'rb') as f:
                reject = pickle.load(f)  

            # Select event_id appropriate
            for run in RUN_LIST :
                print("=> Process run :", run)

                if task == 'LaughterPassive' :
                    if run == '02' or run == '03' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 
                                    'EnvPosed' : 22}
                    elif run == '04' or run == '05' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12,'ScraReal' : 31, 
                                    'ScraPosed' : 32,}
                if task == 'LaughterActive' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66}


                # Take raw data filtered : i.e. NO ICA 
                print('ICA')
                _, raw_path = get_bids_file(PREPROC_PATH, stage='proc-filt_raw', task=task, run=run, subj=subj)
                raw = mne.io.read_raw_fif(raw_path, preload=True)
                raw_filter = raw.copy()
                raw_filter = ica.apply(raw_filter)

                freq_name = FREQS_NAMES[idx]

                print('Apply filter')
                info = raw_filter.info
                raw_filter.filter(l_freq, h_freq) # Apply filter of interest
                raw_hilbert = raw_filter.apply_hilbert(envelope=True)

                picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)

                power_hilbert = raw_hilbert.copy()
                power_hilbert._data = power_hilbert._data**2

                # Segmentation
                print('Segmentation')
                events = mne.find_events(raw)
                epochs = mne.Epochs(
                    power_hilbert,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(None, 0),
                    picks=picks)    

                # Auto-reject epochs
                epochs.drop_bad(reject=reject)

                print(subj_path)
                psd_filename, psd_path = get_bids_file(subj_path, 
                                            stage='hilbert_env-psd_epo', 
                                            subj=subj, 
                                            task=task, 
                                            run=run, 
                                            measure=freq_name)
                epochs.save(subj_path + '/' + psd_filename)

                
        idx += 1

    return epochs

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
    tmin = args.tmin
    tmax = args.tmax

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
        
    epochs_psd = compute_hilbert_psd(SUBJ_CLEAN, RUN_LIST, task=task, FREQS_LIST=FREQS_LIST, tmin=tmin, tmax=tmax)
    
