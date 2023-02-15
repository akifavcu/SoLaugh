import mne
import os
import scipy.stats
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, RESULT_PATH, EVENTS_ID
from mne.stats import spatio_temporal_cluster_test, combine_adjacency

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

# Will be moved in another file later
def ERP(PREPROC_PATH, subj_list, task, cond1, cond2, stage) :
    
    epochs_concat = []
    epochs_all_list = []

    for subj in subj_list :
        _, epochs_file = get_bids_file(PREPROC_PATH, task = task, stage = stage, subj = subj)
        epochs = mne.read_epochs(epochs_file)
        epochs = mne.Epochs.apply_baseline(epochs,baseline=(None,0))
        
        # Average each condition
        condition1 = epochs[cond1].average()
        condition2 = epochs[cond2].average()
        
        # Set manually head mouvement 
        if subj == subj_list[0]:
            head_info = epochs.info['dev_head_t']
        else:
            epochs.info['dev_head_t'] = head_info
        
        epochs_concat = mne.concatenate_epochs([epochs])
        epochs_all_list.append(epochs_concat)
        print('processing -->', subj)

    epochs_save_file = mne.concatenate_epochs(epochs_all_list)

    conditions = {cond1 : condition1, cond2: condition2}

    # Save average condition into pkl file
    for one_condition in conditions :
        _, save_erp = get_bids_file(RESULT_PATH, stage = "erp", condition=one_condition)
        with open(save_erp, 'wb') as f:
            pickle.dump(conditions.get(one_condition), f)
    
    # Save concat condition into pkl file
    conditions = cond1 + "-" + cond2
    _, save_erp_concat = get_bids_file(RESULT_PATH, stage = "erp-concat", condition=conditions)
    
    with open(save_erp_concat, 'wb') as f:
        pickle.dump(epochs_save_file, f)    

    return condition1, condition2, epochs_concat

def cluster_ERP(epochs, event_id, cond1, cond2) :

    # Code adapted from :
    # https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html

    # Drop EEg channels and equalize event number
    epochs.equalize_event_counts(event_id)
    epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')

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

    # We are running an F test, so we look at the upper tail
    tail = 1
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
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=100,
                                                threshold=f_thresh, tail=tail,
                                                n_jobs=None, buffer_size=None,
                                                adjacency=adjacency)
    F_obs, clusters, p_values, _ = cluster_stats

    # Save cluster stats to use it later
    conditions = cond1 + "-" + cond2
    _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "erp-clusters", measure="cluster-stats", condition = conditions)
    
    with open(save_cluster_stats, 'wb') as f:
        pickle.dump(cluster_stats, f)  

    return F_obs, clusters, p_values

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = ["01"]
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
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

    # Compute ERPs
    condition1, condition2, epochs_concat = ERP(PREPROC_PATH, subj_list, task, cond1, cond2, "proc-clean_epo")

    # Compute ERP clusters
    F_obs, clusters, p_values = cluster_ERP(epochs_concat, event_id, cond1, cond2)