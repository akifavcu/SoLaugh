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
    default="Laugh/Real",
    type=str,
    help="First condition",
)

parser.add_argument(
    "-cond2",
    "--condition2",
    default="Laugh/Posed",
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

def induced_cluster(task, stage, cond1, cond2, freq_name) :

    conditions = [cond1, cond2] # assert 2 conditions

    for FREQ, fname in enumerate(freq_name) : 
        
        for i, cond in enumerate(condition) :
            print('condition -->', cond)
            list_all_data = []
            list_epochs_ave = []

            for subj in SUBJ_CLEAN :
                print("processing subject -->", subj)
                list_epochs = []
                epochs_time = []

                for run in ACTIVE_RUN :
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
                    epochs_time.append(epochs_ave_time) # len = n_runs

                epochs_ave_runs = np.mean(np.array(epochs_time), axis = 0) # Average across runs
                epochs_ave_event = np.mean(epochs_ave_runs, axis = 0) # Average across epochs
                print(epochs_ave_event.shape)
                list_epochs_ave.append(epochs_ave_event) # Shape (n_chan)

            # Concat subjects
            data_subj = np.array(list_epochs_ave)  # Shape (n_subj, n_chan)
            print(data_subj.shape)
            list_all_data.append(data_subj) # Append per freqs

            # Concat freqs of each cond
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

        # difference entre les 2 conditions pur chaque sujet
        data_contrast = np.subtract(all_data_cond1, all_data_cond2)
        print(data_contrast.shape) # Shape (n_subj, n_chan)

        # Application de cluster permutation

        # TODO : check adjacency
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

        F_obs, clusters, cluster_p_values, H0 = cluster_stats

        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        print("Good clusters: %s" % good_cluster_inds)

        ########## PLOT ########
        plot_psd_cluster(cluster_stats, task, cond1, cond2, freq_name)


def evoked_cluster(task, stage, cond1, cond2, freq_name) : 

    conditions = [cond1, cond2] # assert 2 conditions
    freq_all = {}

    ######### PREPARE DATA #########

    for FREQ, fname in enumerate(freq_name) : 
        list_all_data = []
        list_evoked_ave = []
        evoked_condition1 = []
        evoked_condition2 = []
        evoked_subj = []
        
        for subj in SUBJ_CLEAN :
            print("processing subject -->", subj)
            list_epochs = []
            evoked_time = []
            contrasts_all_subject = []

            for run in ACTIVE_RUN :
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
                # TODO : check to equalize event

                evoked_condition1.append(epochs[cond1].average()) # Average epochs
                evoked_condition2.append(epochs[cond2].average()) # Average epochs

                # Difference between conditions for each run per subject ??
                contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()], weights=[1, -1])
                contrast.pick_types(meg=True, ref_meg=False,  exclude='bads') # Shape (n_chan, n_times)
                contrasts_all_subject.append(contrast)
                
                evoked_ave_time = np.mean(contrast.get_data(), axis = 1) # Shape (n_channels)
                evoked_time.append(evoked_ave_time) # len = n_runs

            evoked_ave_runs = np.mean(np.array(evoked_time), axis = 0) # Average across runs
            list_evoked_ave.append(evoked_ave_runs) # Shape (n_chan)
            
            evoked_contrast_subj = mne.combine_evoked(contrasts_all_subject, 'equal') # Average run per subject
            evoked_subj.append(evoked_contrast_subj)
            
        # Concat subjects
        data_subj = np.array(list_evoked_ave)  # Shape (n_subj, n_chan)
        list_all_data.append(data_subj) # Append per freqs
        
        all_contrast = mne.combine_evoked(evoked_subj, 'equal') # Average across subject
        freq_all[fname] = all_contrast

        all_data = np.array(list_all_data)
        all_data = np.transpose(all_data, [1, 0, 2])
        print(all_data.shape)

    ######## CLUSTERING #########

    # TODO : check adjacency
    print('Computing adjacency.')
    adjacency, ch_names = compute_ch_adjacency(all_contrast.info, ch_type='mag')
    print(adjacency.shape)

    cluster_stats = \
        permutation_cluster_1samp_test(all_data, 
                                adjacency = adjacency, 
                                out_type='indices',
                                n_permutations=1024, 
                                threshold=None, 
                                tail=0,
                                step_down_p=0.05)

    F_obs, clusters, cluster_p_values, H0 = cluster_stats

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Good clusters: %s" % good_cluster_inds)

    ######## PLOT ########

    plot_psd_cluster(cluster_stats, task, cond1, cond2, freq_name)

    return cluster_stats

def plot_psd_cluster(cluster_stats, task, cond1, cond2, freq_name) : 

    ######## PLOT CLUSTER  ########

    conditions = cond1 + '_' + cond2
    F_obs, clusters, cluster_p_values, H0 = cluster_stats

    for i_clu, clu_idx in enumerate(good_cluster_inds):

        path = 'psd/sub-all_run-all_task-{}_cond-{}_meas-Ttest-cluster_freq{}{}.png'.format(task, conditions, freq_name, i_clu)

        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        
        t_map = F_obs[time_inds, ...].mean(axis=0)

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], all_contrast.info, tmin=0)
        f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='bwr',
                            vlim=(np.min, np.max), show=False,
                            colorbar=False, mask_params=dict(markersize=10))
        
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged T-map (-0.5 - 1.5 s)"
        )
        
        plt.savefig(FIG_PATH + path)
        plt.show()

if __name__ == "__main__" :

    cond1 = args.condition1
    cond2 = args.condition2
    task = args.task
    stage = 'psd'
    freq_names = [args.freqs]

    if cluster_type == 'evoked' :
        cluster_stats = evoked_cluster(task, stage, cond1, cond2, freq_name)

    elif cluster_type == 'induced' :
        cluster_stats = induced_cluster(task, stage, cond1, cond2, freq_name)   