import mne
import os
import scipy.stats
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, RESULT_PATH, EVENTS_ID
from mne.stats import spatio_temporal_cluster_test, combine_adjacency, spatio_temporal_cluster_1samp_test
from mne.epochs import equalize_epoch_counts

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

args = parser.parse_args()

def cluster_Ttest(SUBJ_CLEAN, task, event_id, cond1, cond2) :
    
    contrasts_all_subject = []

    for subj in SUBJ_CLEAN :
        print("processing -->", subj)
        _, path_evoked = get_bids_file(BIDS_PATH, task=task, subj=subj, stage="ave")
        evoked = mne.read_evokeds(path_evoked)
        print(evoked)

        # Drop EEg channels and equalize event number
        contrast = mne.combine_evoked([evoked[0], evoked[1]], weights=[1, -1])
        contrasts_all_subject.append(contrast)
        contrasts_all_subject.pick_types(meg=True, ref_meg=False,  exclude='bads')

        # Equalize trial counts to eliminate bias
        # equalize_epoch_counts([evoked_cond1, evoked_cond2])

    evoked_contrast = mne.combine_evoked(contrasts_all_subject, 'equal')

    # Compute adjacency by using _compute_ch_adjacency function
    # as we got 270 channels and not 275 as the CTF275 provide
    print('Computing adjacency.')
    adjacency, ch_names = compute_ch_adjacency(evoked_contrast.info, ch_type='mag')
    print(adjacency.shape)

    # Obtain the data as a 3D matrix and transpose it such that
    # the dimensions are as expected for the cluster permutation test:
    # n_epochs × n_times × n_channels
    X = np.array([contrast.data for constrast in evoked_contrast])
    print([x.shape for x in X])
    X = [np.transpose(x, (0, 2, 1)) for x in X]
    print([x.shape for x in X])

    X = np.asarray(X)
    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    X_bis = np.transpose(X, [2, 1, 0])
    print(X_bis)

    degrees_of_freedom = len(epochs) - 1
    t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)

    # Run the analysis
    print('Clustering.')
    cluster_stats = \
        spatio_temporal_cluster_1samp_test(X, n_permutations=1024,
                                    threshold=t_thresh, tail=0,
                                    adjacency=adjacency,
                                    out_type='mask', verbose=True)

    T_obs, clusters, cluster_p_values, H0 = cluster_stats

    # Save cluster stats to use it later
    conditions = cond1 + "-" + cond2

    _, save_contrasts = get_bids_file(RESULT_PATH, stage = "erp-contrast", task=task, condition = conditions)

    _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "erp-clusters", task=task, measure="Ttest-clusters", condition = conditions)
    
    with open(save_contrasts, 'wb') as f:
        pickle.dump(evoked_contrast, f)  
    
    with open(save_cluster_stats, 'wb') as f:
        pickle.dump(cluster_stats, f)  

    return cluster_stats

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = SUBJ_CLEAN
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
    conditions = conditions = cond1 + '-' + cond2
    condition_list = [cond1, cond2]
    event_id = dict()
    picks = "meg" # Select MEG channels
    
    for ev in EVENTS_ID :
        for conds in condition_list :
            if conds not in EVENTS_ID :
                raise Exception("Condition is not an event")
            if conds == ev :
                event_id[conds] = EVENTS_ID[ev]

    print("=> Process task :", task, "for conditions :", cond1, "&", cond2)

    # Compute ERP clusters
    cluster_stats = cluster_Ttest(SUBJ_CLEAN, task, event_id, cond1, cond2)