# Import the FOOOF object
import mne
import argparse
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from autoreject import AutoReject
from src.utils import get_bids_file
from src.params import RESULT_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, EVENTS_ID

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
    "-subj",
    "--subject",
    default='01',
    type=str,
    help="Subject to compute",
)
args = parser.parse_args()

def make_epochs(subj_list, task, run_list, tmin, tmax, event_id) :

    for subj in subj_list :
            
            # Take ica data
            _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
            ica = mne.preprocessing.read_ica(ica_path)

            for run in run_list : 

                # Get path
                _, raw_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, stage='proc-filt_raw', run=run)

                raw = mne.io.read_raw_fif(raw_path, preload=True)

                # Apply ICA
                raw_filter = raw.copy()
                raw_filter = ica.apply(raw_filter)

                # Read raw file 
                events = mne.find_events(raw_filter)

                if run == '02' or run == '03' :
                    event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 'EnvPosed' : 22}
                elif run == '04' or run == '05' :
                    event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'ScraReal' : 31,'ScraPosed' : 32,}

                epochs = mne.Epochs(raw_filter,
                                    events=events,
                                    event_id=event_id,
                                    tmin=tmin,
                                    tmax=tmax, 
                                    preload=True)

                epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')
            
                # Auto-Reject per run 
                ar = AutoReject()
                epochs_clean = ar.fit_transform(epochs)  
                epochs_clean.apply_baseline((None,0)) # apply baseline
                
                print("--> save file")
                _, save_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, run=run, measure='clean-', stage='epo')
                epochs_clean.save(save_path, overwrite=True)
                print("--> Done")


if __name__ == "__main__" :

    task = args.task
    subj_list = [args.subject]
    tmin = -1.5
    tmax = 1.5

    if task == 'LaughterActive':
        run_list = ACTIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']
        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66, }

    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'EnvPosed',
                      'ScraReal', 'ScraPosed']
        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 'ScraReal' : 31, 'EnvPosed' : 22,
            'ScraPosed' : 32,}

    make_epochs(subj_list, task, run_list, tmin, tmax, event_id)