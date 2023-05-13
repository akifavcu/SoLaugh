import mne 
import os
import pickle
import argparse
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, FIG_PATH
from src.params import ACTIVE_RUN, PASSIVE_RUN
from mne.time_frequency import (tfr_morlet, AverageTFR)
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test,  permutation_cluster_test


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
    "--freqs",
    default='alpha',
    type=str,
    help="Freqs to compute",
)

args = parser.parse_args()

if __name__ == "__main__" :

    cond1 = args.condition1
    cond2 = args.condition2
    task = args.task
    stage = 'psd'
    freq_names = [args.freqs]

    condition = [cond1, cond2] # assert 2 conditions

    for i, cond in enumerate(condition) :
        print('condition -->', cond)
        list_all_data = []

        for FREQ, fname in enumerate(freq_names) : 
            print('freq -->', fname)
            list_epochs_ave = []

            for subj in SUBJ_CLEAN:
                print("processing subject -->", subj)
                list_epochs = []
                epochs_time = []

                for run in ACTIVE_RUN:
                    print("processing run -->", run)

                    _, psd_path = get_bids_file(RESULT_PATH, 
                                                stage=stage, 
                                                subj=subj, 
                                                task=task, 
                                                run=run, 
                                                measure=fname)
                    epochs = mne.read_epochs(psd_path, verbose=None)

                    # Appliquer la correction baseline
                    epochs = epochs.apply_baseline(baseline=(None, 0))
                    list_epochs.append(epochs[cond])

                # Need to equalize event count
                mne.epochs.equalize_epoch_counts(list_epochs)

                for epo in list_epochs : # Use epochs with equal n_events

                    # Moyenner chaque epoch dans le temps pour chaque condition
                    epochs_ave_time = np.mean(epo.get_data(), axis = 2) # Shape (n_events, n_channels)
                    epochs_time.append(epochs_ave_time) # len = n_uns

                epochs_ave_runs = np.mean(np.array(epochs_time), axis = 0) # Average across runs
                epochs_ave_event = np.mean(epochs_ave_runs, axis = 0) # Average across epochs
                list_epochs_ave.append(epochs_ave_event) # Shape (n_chan)

            # Concat subjects
            data_subj = np.array(list_epochs_ave)  # Shape (n_subj, n_chan)
            list_all_data.append(data_subj) 

        # Concat freqs fon each cond
        if i == 0 : 
            all_data_cond1 = np.array(list_all_data)
            all_data_cond1 = np.transpose(all_data_cond1, [1, 0, 2])
        elif i == 1 : 
            all_data_cond2 = np.array(list_all_data)
            all_data_cond2 = np.transpose(all_data_cond2, [1, 0, 2])

    print('all data condition 1 :', all_data_cond1.shape)  # Shape (n_subj, n_chan, n_freq)
    print('all data condition 2 :', all_data_cond2.shape)  # Shape (n_subj, n_chan, n_freq)

    print('Computing adjacency.')
    adjacency, ch_names = compute_ch_adjacency(epochs.info, ch_type='mag')
    print(adjacency.shape)

    threshold = 6.0
    F_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([all_data_cond1, all_data_cond2], out_type='indices',
                                n_permutations=1024, threshold=threshold, tail=0)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Good clusters: %s" % good_cluster_inds)