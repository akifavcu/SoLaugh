import mne 
import os
import pickle
import argparse
import scipy.stats
import matplotlib
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, FIG_PATH, ACTIVE_RUN, PASSIVE_RUN
from mne.time_frequency import (tfr_morlet, AverageTFR)
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test,  permutation_cluster_test
from mne.stats import (
    spatio_temporal_cluster_test,
    f_threshold_mway_rm,
    f_mway_rm,
    summarize_clusters_stc,
)

def cluster_ttest_psd(freq_name, condition, SUBJ_CLEAN, run_list) : 

    for FREQ, fname in enumerate(freq_name) : 

        for i, cond in enumerate(condition) :
            print('condition -->', cond)
            list_epochs_ave = []
            list_all_data = []
            
            for subj in SUBJ_CLEAN :
                print("processing subject -->", subj)
                list_epochs = []
                epochs_time = []
                
                for run in run_list :
                    print("processing run -->", run)
                    
                    # Get hilbert filtered data
                    _, psd_path = get_bids_file(RESULT_PATH, 
                                                stage=stage, 
                                                subj=subj, 
                                                task=task, 
                                                run=run, 
                                                measure=fname)
                    epochs = mne.read_epochs(psd_path, verbose=None)
                    # NOTE : ADD AUTOREJECT THRESHOLD

                    # Appliquer la correction baseline
                    epochs = epochs.apply_baseline(baseline=(None, 0))
                    list_epochs.append(epochs[cond])
                    
                # Need to equalize event count
                mne.epochs.equalize_epoch_counts(list_epochs)

                for epo in list_epochs : # Use epochs with equal n_events

                    # Moyenner chaque epoch dans le temps pour chaque condition
                    epochs_ave_time = np.mean(epo.get_data(), axis = 2) # Shape (n_events, n_channels)
                    epochs_time.append(epochs_ave_time) # len = n_runs

                epochs_ave_runs = np.mean(np.array(epochs_time), axis = 0) # Average across runs
                epochs_ave_event = np.mean(epochs_ave_runs, axis = 0) # Average across epochs
                list_epochs_ave.append(epochs_ave_event) # Shape (n_chan)

            # Concat subjects
            data_subj = np.array(list_epochs_ave)  # Shape (n_subj, n_chan)
            list_all_data.append(data_subj) # Append per freqs

            # Concat freqs fon each cond
            if i == 0 : 
                all_data_cond1 = np.array(list_all_data)
                all_data_cond1 = np.transpose(all_data_cond1, [1, 0, 2])
                #all_data_cond1 = data_subj
            elif i == 1 : 
                all_data_cond2 = np.array(list_all_data)
                all_data_cond2 = np.transpose(all_data_cond2, [1, 0, 2])
                #all_data_cond2 = data_subj
            
        print('all data condition 1 :', all_data_cond1.shape)  # Shape (n_subj, n_freq, n_chan)
        print('all data condition 2 :', all_data_cond2.shape)  # Shape (n_subj, n_freq, n_chan)

        data_contrast = np.subtract(all_data_cond1, all_data_cond2)

        print(data_contrast.shape)
        data_contrast = np.subtract(all_data_cond1, all_data_cond2)

        # Application de cluster permutation
        print('Computing adjacency.')
        adjacency, ch_names = compute_ch_adjacency(epochs.info, ch_type='mag')
        print(adjacency.shape)

        pval = 0.01  # arbitrary
        dfn = len([all_data_cond1, all_data_cond2]) - 1  # degrees of freedom numerator
        dfd = len(all_data_cond1) - len([all_data_cond1, all_data_cond2])  # degrees of freedom denominator
        thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution

        cluster_stats = \
            permutation_cluster_1samp_test(data_contrast, 
                                    adjacency = adjacency, 
                                    out_type='indices',
                                    n_permutations=1024, 
                                    threshold=None, 
                                    tail=0,
                                    step_down_p=0.05)

        list_good_cluster_inds.append(cluster_stats)

        # Save cluster stats to use it later
        # TODO : save all subject evoked_cond1 et cond2 
        conditions = cond1 + "-" + cond2

        _, save_contrasts = get_bids_file(RESULT_PATH, stage = "psd-contrast", task=task, condition = conditions)

        _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "psd-clusters", task=task, measure="Ttest-clusters", condition = conditions)

        with open(save_contrasts, 'wb') as f:
            pickle.dump(data_contrast, f)  

        with open(save_cluster_stats, 'wb') as f:
            pickle.dump(cluster_stats, f)  

    return data_contrast, cluster_stats

if __name__ == "__main__" :

    task = 'LaughterActive'
    stage = 'psd_epo'
    cond1 = 'LaughReal'
    cond2 = 'LaughPosed'

    freq_name = FREQS_NAMES
    condition = [cond1, cond2] # assert 2 conditions
    list_good_cluster_inds = []

    if task == 'LaughterActive' : 
        run_list = ACTIVE_RUN

    elif task == 'LaughterPassive' :
        run_list = PASSIVE_RUN

    data_contrast, cluster_stats = cluster_ttest_psd()