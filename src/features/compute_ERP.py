import mne
import os
import scipy.stats
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, RESULT_PATH, EVENTS_ID, SUBJ_CLEAN
from mne.stats import spatio_temporal_cluster_test, combine_adjacency, permutation_cluster_1samp_test

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

def global_field_power(subj_list, task, event_id, cond1, cond2):
    
    data_all_subject = []

    for subj in subj_list :
        epochs_path = get_bids_file(RESULT_PATH, stage='AR_epochs', subj=subj, task=task)
        epochs = mne.read_epochs(epochs_path)

        # Drop EEg channels and equalize event number
        # Equalize event
        epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')

        data_all_subject.append(epochs.get_data())

    # Average across subjects
    data = np.mean(np.array(data_all_subject), axis=0) 

    # Average across epochs

    # Average across channel

    # Test-t through time (correction permutation)


    
def cluster_ERP(subj_list, task, event_id, cond1, cond2) :

    data_all_subject = []

    # Code adapted from :
    # https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html

    for subj in subj_list :
        epochs_path = get_bids_file(RESULT_PATH, stage='AR_epochs', subj=subj, task=task)
        epochs = mne.read_epochs(epochs_path)

        # Drop EEg channels and equalize event number
        epochs.equalize_event_counts(event_id)
        epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')

        data_all_subject.append(epochs.get_data())

    # Compute adjacency by using _compute_ch_adjacency function
    # as we got 270 channels and not 275 as the CTF275 provide
    adjacency, ch_names = compute_ch_adjacency(epochs.info, ch_type='mag')
    print(adjacency.shape)

    # Obtain the data as a 3D matrix and transpose it such that
    # the dimensions are as expected for the cluster permutation test:
    # n_epochs × n_times × n_channels
    X = [epochs[event_name].get_data() for event_name in event_id]
    print([x.shape for x in X])
    X = [np.transpose(x, (0, 2, 1)) for x in X]
    print([x.shape for x in X])

    X = np.asarray(X)

    # We are running an F test, so we look at the upper tail
    tail = 0
    alpha_cluster_forming = 0.01

    # For an F test we need the degrees of freedom for the numerator
    # (number of conditions - 1) and the denominator (number of observations
    # - number of conditions):
    n_conditions = len(event_id)
    n_observations = len(X[0])
    dfn = n_conditions - 1
    dfd = n_observations - n_conditions

    # Note: we calculate 1 - alpha_cluster_forming to get the critical value
    # on the right tail
    f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)

    # run the cluster based permutation analysis
    cluster_stats =  spatio_temporal_cluster_test(X, n_permutations=1000,
                                                threshold=f_thresh, tail=tail,
                                                n_jobs=None, buffer_size=None,
                                                adjacency=adjacency)
    F_obs, clusters, p_values, _ = cluster_stats

    # Save cluster stats to use it later
    conditions = cond1 + "-" + cond2
    _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "erp-clusters", task=task, measure="cluster-stats", condition = conditions)
    
    with open(save_cluster_stats, 'wb') as f:
        pickle.dump(cluster_stats, f)  

    return F_obs, clusters, p_values

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = SUBJ_CLEAN
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
    conditions = cond1 + '-' + cond2
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

    # Compute GFP
    global_field_power(subj_list, task, event_id, cond1, cond2)

    # Compute ERP clusters
    # F_obs, clusters, p_values = cluster_ERP(subj_list, task, event_id, cond1, cond2)
