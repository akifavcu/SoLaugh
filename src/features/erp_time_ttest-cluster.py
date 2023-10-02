import mne
import os
import scipy.stats
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, RESULT_PATH, EVENTS_ID, FIG_PATH
from mne.stats import spatio_temporal_cluster_test, combine_adjacency, spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test
from mne.epochs import equalize_epoch_counts
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_compare_evokeds

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
    help="Condition 1 to compute",
)
parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Condition 2 to compute",
)
parser.add_argument(
    "-cond3",
    "--condition3",
    default="Miss",
    type=str,
    help="Condition 3 to compute",
)

args = parser.parse_args()

def prepare_data_ttest_cluster(subj_list, task, cond1, cond2, ROI, button_cond) : 

    for roi in ROI : 
        
        contrasts_all_subject = []
        contrasts_all_subject_all_chan = []

        evoked_condition1 = []
        evoked_condition2 = []

        evoked_condition1_data = []
        evoked_condition2_data = []

        conditions = cond1 + '-' + cond2
        for subj in subj_list :
            print("processing -->", subj)
            
            chan_selected = []
            
            # TODO : change with AR_epochs
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            epochs.apply_baseline(baseline=(None, 0))
            
            CHAN = epochs.info['ch_names']
            
            for chan in CHAN  : 
                if roi in chan : 
                    chan_selected.append(chan)
            
            epochs_copy = epochs.copy() 
            epochs_copy.pick_channels(chan_selected)
                    
            if button_cond == True : # Compute button press : Combine events Good & Bad
                evoked_cond1_roi = mne.combine_evoked([epochs_copy[cond1].average(), epochs_copy[cond3].average()], weights='nave')
                evoked_cond1_all_chan = mne.combine_evoked([epochs[cond1].average(), epochs[cond3].average()], weights='nave')

            else : # For other conditions
                evoked_cond1_roi = epochs_copy[cond1].average()
                evoked_cond1_all_chan = epochs[cond1].average()
                
            # Create fake evoked with value=0 
            if cond2 == 'BaselineZero' : 
                baseline_data = np.zeros((epochs_copy.get_data().shape[1], epochs_copy.get_data().shape[2]))
                evoked_cond2_roi = mne.EvokedArray(baseline_data, epochs_copy.info, tmin=-0.5, comment='baseline')

                baseline_data_all_chan = np.zeros((epochs.get_data().shape[1], epochs.get_data().shape[2]))
                evoked_cond2_all_chan = mne.EvokedArray(baseline_data_all_chan, epochs.info, tmin=-0.5, comment='baseline')
            else :
                evoked_cond2_roi = epochs_copy[cond2].average()
                evoked_cond2_all_chan = epochs[cond2].average()

            
            # Prepare data for ttest with roi
            evoked_condition2_data.append(evoked_cond2_roi.get_data()) 
            evoked_condition1_data.append(evoked_cond1_roi.get_data())
            
            evoked_condition2.append(evoked_cond2_roi) 
            evoked_condition1.append(evoked_cond1_roi)

            contrast = mne.combine_evoked([evoked_cond1_roi, evoked_cond2_roi], weights=[1, -1])
            contrast_all_chan = mne.combine_evoked([evoked_cond1_all_chan, evoked_cond2_all_chan], weights=[1, -1])
            contrast.pick_types(meg=True, ref_meg=False,  exclude='bads')
            
            contrasts_all_subject.append(contrast)
            contrasts_all_subject_all_chan.append(contrast_all_chan)

        # Combine all subject together
        evokeds = {cond1 : evoked_condition1, cond2 : evoked_condition2}

        evoked_contrast = mne.combine_evoked(contrasts_all_subject, 'equal')
        evoked_contrast_all_chan = mne.combine_evoked(contrasts_all_subject_all_chan, 'equal')

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

        X = np.mean(X, axis=2)
        print(X.shape)

        degrees_of_freedom = len(contrasts_all_subject) - 1
        t_thresh = scipy.stats.t.ppf(1 - 0.01 / 2, df=degrees_of_freedom)

        # Run the analysis
        print('Clustering.')
        cluster_stats = \
            permutation_cluster_1samp_test(X, n_permutations=1024,
                                        threshold=None, tail=0,
                                        out_type='indices', verbose=None, 
                                        step_down_p = 0.05, check_disjoint=True)

        T_obs, clusters, cluster_p_values, H0 = cluster_stats

        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        print("Good clusters: %s" % good_cluster_inds)

        plot_cluster_time(task, roi, conditions, cluster_stats, evoked_contrast, evoked_contrast_all_chan, evokeds)

def plot_cluster_time(task, roi, conditions, cluster_stats, evoked_contrast, evoked_contrast_all_chan, evokeds): 
    print('Plot')
    T_obs, clusters, p_values, _ = cluster_stats

    p_accept = 0.05
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # configure variables for visualization
    colors = "r",'steelblue'
    linestyles = '-', '--'

    times = evoked_contrast.times * 1e3

    su = 'all'
    task = task
    run = 'all'
    cond = conditions
    meas = 'ttest-cluster'
    stage= roi + '-erp'

    save_fig_path = FIG_PATH + '/erp/ttest_time_cluster/sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}.png'.format(su,
                                                                                                            task,
                                                                                                            run,
                                                                                                            cond,
                                                                                                            meas,
                                                                                                            stage)

    # Prepare mask for topomaps
    ch_inds = []

    for i, chan in enumerate(evoked_contrast_all_chan.info['ch_names']):
        if roi in evoked_contrast_all_chan.info['ch_names'][i] :    
            ch_inds.append(i)
            
    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        
        evoked_contrast_topo = evoked_contrast_all_chan.copy()

        # unpack cluster information, get unique indices
        time_inds = clusters[clu_idx]
        time_inds = np.unique(time_inds)
        sig_times = evoked_contrast.times[time_inds]
        
        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(17, 5))
        
        evoked_contrast_topo.crop(tmin=sig_times[0], tmax=sig_times[-1])
        
        # create spatial mask
        mask = np.zeros((evoked_contrast_all_chan.get_data().shape[0], evoked_contrast_topo.times.shape[0]), dtype=bool)
        mask[ch_inds, :] = True

        
        evoked_contrast_topo.plot_topomap(times='peaks', mask=mask, axes=ax_topo, cmap='bwr',
                            cnorm = matplotlib.colors.CenteredNorm(vcenter=0), show=False,
                            colorbar=False, mask_params=dict(markersize=10), extrapolate='head',
                            sphere=(0, 0.019, 0, 0.184))

        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)

        title = 'Cluster #{0}, (p={1}), ROI {2} - {3}'.format(i_clu + 1, p_values[clu_idx], 
                                                              roi, cond)

        plot_compare_evokeds(evokeds, title=title,
                        colors=colors, show=False,axes=ax_signals,
                        split_legend=True, truncate_yaxis='auto', combine="mean")

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                            color='orange', alpha=0.3)
        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)

        print('Save file')
        plt.savefig(save_fig_path)
        
        del evoked_contrast_topo

if __name__ == "__main__" :

    task = args.task
    subj_list = SUBJ_CLEAN

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
    cond3 = args.condition3

    if cond1 == 'Good' and 'cond3' == 'Miss' :
        button_cond = True
    else :
        button_cond = False
            
    ROI = ['MLT', 'MRT', 'MLP', 'MRP', 'MLC',
            'MRC', 'MLF', 'MRF','MLO','MRO']

    prepare_data_ttest_cluster(subj_list, task, cond1, cond2, ROI, button_cond)