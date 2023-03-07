import mne
import os
import scipy.stats
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from src.utils import get_bids_file
from src.params import RESULT_PATH, SUBJ_CLEAN, ACTIVE_RUN, FIG_PATH, EVENTS_ID
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

def plot_ERP(condition1, condition2, cond1, cond2, task, picks) :
    
    condition1.pick_types(meg=True, ref_meg = False,  exclude='bads')
    condition2.pick_types(meg=True, ref_meg = False,  exclude='bads')

    condition1.filter(l_freq=1, h_freq=30)
    condition1.resample(sfreq=300)
    
    condition2.filter(l_freq=1, h_freq=30)
    condition2.resample(sfreq=300)
    
    # Plot ERPs LaughReal vs LaughPosed
    evoked = dict(real = list(condition1.iter_evoked()), posed = list(condition2.iter_evoked()))
    mne.viz.plot_compare_evokeds(evoked, picks = picks, combine='mean', time_unit='ms')
    fname_ERP = FIG_PATH + "erp/subj-all_run-all_task-{}_plot-erp_cond-{}-{}".format(task, cond1, cond2)
    plt.savefig(fname_ERP)

    # Average each condition
    condition1 = condition1.average()
    condition2 = condition2.average()

    # Plot each condition separately
    fig_cond1 = condition1.plot_joint(picks = picks)    
    fname_cond1 = FIG_PATH + "erp/subj-all_run-all_task-{}_plot-erp_cond-{}".format(task, cond1)
    fig_cond1.savefig(fname_cond1)

    fig_cond2 = condition2.plot_joint(picks = picks)
    fname_cond2 = FIG_PATH + "erp/subj-all_run-all_task-{}_plot-erp_cond-{}".format(task, cond2)
    fig_cond2.savefig(fname_cond2)
    


def visualize_cluster(epochs, cluster_stats, event_id, task, conditions, cond1, cond2) :
    """
    Code adapted from : 
    https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html
    """

    #epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')
    T_obs, clusters, p_values, _ = cluster_stats
    
    p_accept = 0.01
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # configure variables for visualization
    colors = "crimson",'steelblue'
    linestyles = '-', '--'

    # organize data for plotting
    epochs.pick_types(meg=True, ref_meg = False,  exclude='bads')
    epochs.filter(l_freq=1, h_freq=30)
    evokeds = {cond: epochs[cond].average() for cond in event_id}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = T_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = epochs.times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], epochs.info, tmin=0)
        t_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                              vlim=(np.min, np.max), show=False,
                              colorbar=False, mask_params=dict(markersize=10))

        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)

        # TODO add color for noise around signal
        title = 'Cluster #{0}, {1} sensor(s) (p < {2})'.format(i_clu + 1, len(ch_inds), p_values[clu_idx])
        if len(ch_inds) > 1:
            title += "(mean)"

        plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, show=False,
                         split_legend=True, truncate_yaxis='auto')

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.savefig(FIG_PATH + 'erp/sub-all_run-all_task-{}_cond-{}_meas-Ttest-cluster_erp{}.png'.format(task, conditions, i_clu))

if __name__ == "__main__" :

    # Conditions and task to compute
    task = args.task
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

    # Import ERP files path
    _, save_erp_cond1 = get_bids_file(RESULT_PATH, task=task, stage="erp", condition=cond1)

    _, save_erp_cond2 = get_bids_file(RESULT_PATH, task=task, stage="erp", condition=cond2)

    _, save_erp_concat = get_bids_file(RESULT_PATH, task=task, stage="erp-concat", condition=conditions)

    _, save_clusters_stats = get_bids_file(RESULT_PATH, stage="erp-clusters", task=task, measure="Ttest-clusters", condition=conditions)

    # Open pickle files
    with open(save_erp_cond1, "rb") as f:
        condition1 = pickle.load(f)

    with open(save_erp_cond2, "rb") as f:
        condition2 = pickle.load(f)

    with open(save_erp_concat, "rb") as f:
        epochs_concat = pickle.load(f)

    with open(save_clusters_stats, 'rb') as f:
        cluster_stats = pickle.load(f)

    # Plot ERPs
    #plot_ERP(condition1, condition2, cond1, cond2, task, picks)

    # Visualization of ERP clusters
    visualize_cluster(epochs_concat, cluster_stats, event_id, task, conditions, cond1, cond2)
