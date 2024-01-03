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
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test


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

def cluster_ttest_psd(freq_name, condition, subj_list, run_list, stage, task) : 

    for FREQ, fname in enumerate(freq_name) : 

        contrasts_all_subject = []
        evoked_condition1 = []
        evoked_condition2 = []
    
        for i_subj, subj in enumerate(subj_list):
            print("=> Process task :", task, 'subject', subj)

            epochs_list = []
            evoked_condition1_data = []
            evoked_condition2_data = []

            sub_id = 'sub-' + subj
            subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id, 'hilbert/')

            if not os.path.isdir(subj_path):
                os.mkdir(subj_path)
                print('BEHAV folder created at : {}'.format(subj_path))
            else:
                print('{} already exists.'.format(subj_path))

            # Select event_id appropriate
            for run in run_list :
                print("=> Process run :", run)

                psd_filename, psd_path = get_bids_file(subj_path, 
                                        stage='hilbert_env-psd_epo', 
                                        subj=subj, 
                                        task=task, 
                                        run=run, 
                                        measure=fname)

                epochs = mne.read_epochs(subj_path + psd_filename)
                
                if subj == subj_list[0] and run == run_list[0]:
                    head_info = epochs.info['dev_head_t']
                else:
                    epochs.info['dev_head_t'] = head_info

                epochs.equalize_event_counts([cond1, cond2])

                epochs_list.append(epochs)

                del epochs

            epochs_all_run = mne.concatenate_epochs(epochs_list)  

            # Prepare data for ttest with roi
            evoked_condition1_data.append(epochs_all_run[cond1].average().get_data()) 
            evoked_condition1.append(epochs_all_run[cond1].average()) 

            if cond2 == 'BaselineZero' : 
                baseline_data = np.zeros((epochs_all_run.get_data().shape[1], epochs_all_run.get_data().shape[2]))
                evoked_cond2_roi = mne.EvokedArray(baseline_data, epochs_all_run.info, tmin=-0.5, comment='baseline')
                
            else : 
                evoked_condition2_data.append(epochs_all_run[cond2].average().get_data())
                evoked_condition2.append(epochs_all_run[cond2].average())

            contrast = mne.combine_evoked([epochs_all_run[cond1].average(), epochs_all_run[cond2].average()], weights=[1, -1])
            contrast.pick_types(meg=True, ref_meg=False,  exclude='bads')

            contrasts_all_subject.append(contrast)

        # Application de cluster permutation
        evokeds = {cond1 : evoked_condition1, cond2 : evoked_condition2}
            
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
                                        threshold=None, tail=0,
                                        adjacency=adjacency,
                                        out_type='indices', verbose=None, 
                                        step_down_p = 0.05, check_disjoint=True)

        T_obs, clusters, cluster_p_values, H0 = cluster_stats

        good_cluster_inds = np.where(cluster_p_values < 0.01)[0]
        print("Good clusters: %s" % good_cluster_inds)

        # Save cluster stats to use it later
        # TODO : save all subject evoked_cond1 et cond2 
        conditions = cond1 + "-" + cond2 + '-' + fname

        _, save_contrasts = get_bids_file(RESULT_PATH, stage = "psd-contrast", task=task, condition = conditions)

        _, save_cluster_stats = get_bids_file(RESULT_PATH, stage = "psd-clusters", task=task, measure="Ttest-clusters", condition = conditions)

        with open(save_contrasts, 'wb') as f:
            pickle.dump(evoked_contrast, f)  

        with open(save_cluster_stats, 'wb') as f:
            pickle.dump(cluster_stats, f)  

    return evoked_contrast, cluster_stats

if __name__ == "__main__" :

    task = args.task
    stage = 'psd_epo'
    cond1 = args.condition1
    cond2 = args.condition2

    freq_name = FREQS_NAMES
    condition = [cond1, cond2] # assert 2 conditions
    list_good_cluster_inds = []
    subj_list = SUBJ_CLEAN

    if task == 'LaughterActive' : 
        run_list = ACTIVE_RUN

    elif task == 'LaughterPassive' :
        run_list = PASSIVE_RUN

    data_contrast, cluster_stats = cluster_ttest_psd(freq_name, condition, subj_list, run_list, stage, task)