import mne
import os
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, RESULT_PATH
from mne.stats import spatio_temporal_cluster_test, combine_adjacency

# Will be moved in another file later
def ERP(PREPROC_PATH, subj_list, run_list, cond1, cond2, stage) :
    
    epochs_concat = []

    for subj in subj_list :
        for run in run_list :
            epochs_file = get_bids_file(PREPROC_PATH, subj, run, stage)
            epochs = mne.read_epochs(epochs_file)
            print(epochs.info)

            # Average each condition
            condition1 = epochs[cond1].average()
            condition2 = epochs[cond2].average()

            epochs_concat = mne.concatenate_epochs([epochs]) # See if problem with head location

    # Need to save epochs_concat
    if len(subj_list) == 32 :
        subj_name = "all"
    else :
        subj_name = "notall"

    # TODO : Need to put that in params
    save_cond1 = RESULT_PATH + "meg/reports/epochs/ave-epo_cond-{}_epochs.pkl".format(cond1)
    save_cond2 = RESULT_PATH + "meg/reports/epochs/ave-epo_cond-{}_epochs.pkl".format(cond2)
    save_concat = RESULT_PATH + "meg/reports/epochs/concat-epo_cond-{}-{}_epochs.pkl".format(cond1, cond2)

    with open(save_cond1, 'wb') as f:
        pickle.dump(condition1, f)

    with open(save_cond2, 'wb') as f:
        pickle.dump(condition2, f)
    
    with open(save_concat, 'wb') as f:
        pickle.dump(epochs_concat, f)    

    return condition1, condition2, epochs_concat

def cluster_ERP(epochs, event_id) :

    # Code adapted from :
    # https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html
    
    # Drop EEg channels and equalize event number
    epochs.drop_channels(['EEG057','EEG058','EEG059'])
    epochs.equalize_event_counts(event_id)

    # Obtain the data as a 3D matrix and transpose it such that
    # the dimensions are as expected for the cluster permutation test:
    # n_epochs × n_times × n_channels
    X = [epochs[event_name].get_data() for event_name in event_id]
    X = [np.transpose(x, (0, 2, 1)) for x in X]

    # We are running an F test, so we look at the upper tail
    tail = 1
    alpha_cluster_forming = 0.001

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
                                                adjacency=None)
    F_obs, clusters, p_values, _ = cluster_stats

    # TODO : Need to find a better way to do this save !
    save_Fobs = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-Fobs_pval-{}.pkl".format(alpha_cluster_forming)
    save_clusters = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-clusters_pval-{}.pkl".format(alpha_cluster_forming)
    save_pval = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-pva_pval-{}.pkl".format(alpha_cluster_forming)

    with open(save_Fobs, 'wb') as f:
        pickle.dump(F_obs, f)

    with open(save_clusters, 'wb') as f:
        pickle.dump(clusters, f)
    
    with open(save_pval, 'wb') as f:
        pickle.dump(p_values, f)
        
    return F_obs, clusters, p_values

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    subj_list = ["01", "02"]
    run_list = ["07"]
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = "LaughReal"
    cond2 = "LaughPosed"
    picks = "meg" # Select MEG channels
    event_id = {'LaughReal' : 11, 'LaughPosed' : 12}

    # Compute ERPs
    condition1, condition2, epochs_concat = ERP(PREPROC_PATH, subj_list, run_list, cond1, cond2, "epo")

    # Compute ERP clusters
    F_obs, clusters, p_values = cluster_ERP(epochs_concat, event_id)