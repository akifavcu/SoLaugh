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

def prepare_psd_data(conditions, SUBJ_CLEAN, task):
    all_subj_psd = []

    for i, cond in enumerate(conditions) :
        print('condition -->', cond)
        list_all_data = []
        list_epochs_ave = []

        for subj in SUBJ_CLEAN :
            epochs_time = []
            list_epochs = []

            for run in run_list : 
                _, psd_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, run=run, stage='proc-clean_epo')
                epochs = mne.read_epochs(psd_path)
                epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')
                epochs = epochs.apply_baseline(baseline=(None, 0))
                
                psd = epochs[cond].compute_psd(fmin=2.0, fmax=40.0)

                freqs_subj = psd.freqs
                freqs_list.append(freqs_subj)
                
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

    _, save_path = get_bids_file(RESULT_PATH, stage='psd', measure = 'log_fooof', task=task, condition=cond)

    with open(save_path, 'wb') as f:
        pickle.dump(psd, f)  

    return all_subj_psd

def apply_fooof(data_psd) :
    fm = FOOOF()

    # Set the frequency range to fit the model
    freq_range = [2, 40]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fooof_path = os.path.join(RESULT_PATH, 'meg', 'reports', 'sub-all')
    fm.fit(data_psd, data_psd, freq_range)  

    fm.save('fooof_data', fooof_path, save_results=True, save_settings=True, save_data=True)
    fm.save_report('fooof_report', fooof_path)

if __name__ == "__main__" :
    task = 'LaughterActive'
    conditions = ['LaughReal']

    if task == 'LaughterActive':
        run_list = ACTIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'EnvPosed'
                      'ScraReal', 'ScraPosed']

    data_psd = prepare_psd_data(conditions, SUBJ_CLEAN, task)