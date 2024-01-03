import os
import pickle
import argparse

from src.utils import get_bids_file, compute_ch_adjacency
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, RESULT_PATH, EVENTS_ID
from src.arg_parserarg_parser import get_task_condition_parser

import mne
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from mne.stats import spatio_temporal_cluster_1samp_test

# Need to check if that works
parser = get_task_condition_parser()

parser.add_argument(
    "-bsl",
    "--baseline",
    default=False,
    type=bool,
    help="Compute baseline condition",
)
args = parser.parse_args()

def compute_cluster_Ttest(SUBJ_CLEAN, task, cond1, cond2, bsl) :
    
    contrasts_all_subject = []
    evoked_condition1 = []
    evoked_condition2 = []

    for subj in SUBJ_CLEAN :

        if bsl == False : 
            print("processing -->", subj)
            # TODO : change with AR_epochs
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            epochs.equalize_event_counts([cond1, cond2]) 

            evoked_cond1 = epochs[cond1].average()
            evoked_cond2 = epochs[cond2].average()

            # Drop EEg channels and equalize event number
            evoked_condition1.append(evoked_cond1) 
            evoked_condition2.append(evoked_cond2)

            contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()], weights=[1, -1])
            contrast.pick_types(meg=True, ref_meg=False,  exclude='bads')
            contrasts_all_subject.append(contrast)

        elif bsl == True :

            cond2 = 'BaselineZero'
            conditions = cond1 + '-' + cond2
            condition_list = [cond1, cond2]

            print("processing -->", subj)
            # TODO : change with AR_epochs per run
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            
            evoked_cond1 = epochs[cond1].average()
            evoked_cond1.apply_baseline((None, 0))

            # Take nb channel and time
            baseline_data = np.zeros((epochs.get_data().shape[1], epochs.get_data().shape[2]))

            # Drop EEg channels and equalize event number
            baseline = mne.EvokedArray(baseline_data, epochs.info, tmin=-0.5, comment='baseline')

            contrast = mne.combine_evoked([evoked_cond1, baseline], weights=[1, -1])
            contrast.pick_types(meg=True, ref_meg=False,  exclude='bads')
            contrasts_all_subject.append(contrast)

    # Combine all subject together
    evoked_contrast = mne.combine_evoked(contrasts_all_subject, 'equal')

    # Compute adjacency by using compute_ch_adjacency function
    # as we have 270 channels and not 275 as the CTF275 template provide
    print('Computing adjacency.')
    adjacency, ch_names = compute_ch_adjacency(evoked_contrast.info, ch_type='mag')
    print(adjacency.shape)

    # Obtain the data as a 3D matrix and transpose it such that
    # the dimensions are as expected for the cluster permutation test:
    # n_epochs × n_times × n_channels
    X = np.array([c.data for c in contrasts_all_subject])
    X = np.transpose(X, [0, 2, 1])
    print(X.shape)

    degrees_of_freedom = len(contrasts_all_subject) - 1
    t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)

    # Run the analysis
    print('Clustering.')
    cluster_stats = \
        spatio_temporal_cluster_1samp_test(X, n_permutations=1024,
                                    threshold=t_thresh, tail=0,
                                    adjacency=adjacency,
                                    out_type='indices', verbose=None, 
                                    step_down_p = 0.05, check_disjoint=True)

    T_obs, clusters, cluster_p_values, H0 = cluster_stats

    good_cluster_inds = np.where(cluster_p_values < 0.01)[0]
    print("Good clusters: %s" % good_cluster_inds)

    # Save cluster stats to use it later
    conditions = cond1 + "-" + cond2

    _, save_contrasts = get_bids_file(RESULT_PATH, stage = "erp-contrast-apply-baseline", task=task, condition = conditions)

    _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "erp-clusters-apply-baseline", task=task, measure="Ttest-clusters", condition = conditions)

    with open(save_contrasts, 'wb') as f:
        pickle.dump(contrast, f)  
    
    with open(save_cluster_stats, 'wb') as f:
        pickle.dump(cluster_stats, f)  

    return cluster_stats, contrast, evoked_contrast, evoked_condition1, evoked_condition2

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = SUBJ_CLEAN

    # Select what conditions to compute (str)
    bsl = args.baseline
    cond1 = args.condition1
    cond2 = args.condition2
    conditions = cond1 + '-' + cond2
    condition_list = [cond1, cond2]
    
    print("=> Process task :", task, "for conditions :", cond1, "&", cond2)

    # Compute ERP clusters
    cluster_stats, contrast, evoked_contrast, evoked_condition1, evoked_condition2 = compute_cluster_Ttest(SUBJ_CLEAN, task, cond1, cond2, bsl)