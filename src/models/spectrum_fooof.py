# Import the FOOOF object
import mne
import pickle
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
from src.utils import get_bids_file
from src.params import RESULT_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
args = parser.parse_args()

def prepare_psd_data(SUBJ_CLEAN, task, conditions):
    all_subj_psd = []

    for i, cond in enumerate(conditions) :
        print('condition -->', cond)
        list_all_data = []
        list_epochs_ave = []
        all_psd = []
        freqs_list = []

        for subj in SUBJ_CLEAN :
            epochs_time = []
            list_epochs = []

            for run in run_list : 
                # Get path
                _, psd_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, run=run, stage='proc-clean_epo')

                # Read epochs
                epochs = mne.read_epochs(psd_path)
                epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')
                epochs = epochs.apply_baseline(baseline=(None, 0))
                
                # Compute PSD
                psd = epochs[cond].compute_psd(fmin=2.0, fmax=40.0)

                # Take freqs
                freqs_subj = psd.freqs
                freqs_list.append(freqs_subj)
                
                # Take data
                data = psd.get_data()
                all_psd.append(data)
                data_ave_epochs = np.mean(data, axis = 0)

        # Average frequency across subjects
        freqs = np.mean(np.array(freqs_list), axis=0)
        print(freqs.shape)

        # Average power across :
        # runs
        psd_ave_run = np.mean(np.array(all_psd), axis=0)
        # epochs
        psd_ave_epo = np.mean(np.array(psd_ave_run), axis=0)
        # channels
        psd = np.mean(np.array(psd_ave_epo), axis=0)

        print(psd.shape)

        _, save_path = get_bids_file(RESULT_PATH, stage='psd', measure='log_fooof', task=task, condition=cond)

        with open(save_path, 'wb') as f:
            pickle.dump(psd, f)  
        
        apply_fooof(psd, task=task, condition=cond)

    return psd

def apply_fooof(psd, task, condition) :

    fooof_path = os.path.join(RESULT_PATH, 'meg', 'reports', 'sub-all')

    # Fit model to data
    fm = FOOOF()
    fm.fit(freqs, psd, freq_range)  

    # Save data
    fm.save('subj-all_task-{}_cond-{}_fooof_data'.format(task, condition), 
            fooof_path, save_results=True, save_settings=True, save_data=True)

    # Save fooof report
    fm.save_report('subj-all_task-{}_cond-{}_fooof_report'.format(task, condition), fooof_path)

if __name__ == "__main__" :
    task = args.task

    if task == 'LaughterActive':
        run_list = ACTIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'EnvPosed'
                      'ScraReal', 'ScraPosed']

    psd = prepare_psd_data(SUBJ_CLEAN, task, conditions)