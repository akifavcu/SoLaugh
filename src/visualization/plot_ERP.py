import mne
import os
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.utils import get_bids_file
from src.params import RESULT_PATH, SUBJ_LIST, ACTIVE_RUN, FIG_PATH
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.viz import plot_compare_evokeds

# Remove plot for compute canada
def plot_ERP(condition1, condition2, cond1_name, cond2_name, picks) :

    # Plot each condition separately
    fig_cond1 = condition1.plot_joint(picks = picks)
    fname_cond1 = FIG_PATH + "plot-join_cond-" + cond1_name
    fig_cond1.savefig(fname_cond1)

    fig_cond2 = condition2.plot_joint(picks = picks)
    fname_cond2 = FIG_PATH + "plot-join_cond-" + cond2_name
    fig_cond2.savefig(fname_cond2)
    
    # Plot ERPs LaughReal vs LaughPosed
    evoked = dict(real = condition1, posed = condition2)
    mne.viz.plot_compare_evokeds(evoked, picks = picks, combine='mean')
    fname_ERP = FIG_PATH + "ERP_cond1-" + cond1_name + "_cond2-" + cond2_name
    # fig_ERP.savefig(fname_ERP) # Doesn't work


def visualize_cluster(epochs, F_obs, clusters, p_values, event_id) :

    # Code adapted from :
    # https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html
    
    p_accept = 0.01
    good_cluster_inds = np.where(p_values < p_accept)[0]

    # configure variables for visualization
    colors = {"LaughReal": "crimson", "LaughPosed": 'steelblue'}

    # organize data for plotting
    evokeds = {cond: epochs[cond].average() for cond in event_id}

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        print(ch_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        f_map = F_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = epochs.times[time_inds]

        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], epochs.info, tmin=0)
        f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
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
            'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"
        plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
                            colors=colors, linestyles=None, show=False,
                            split_legend=True, truncate_yaxis='auto')
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                color='orange', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        fig.savefig(FIG_PATH + "plot-clusters_ERP")
        plt.show()
        plt.close()
        "Done Visualize cluster"

if __name__ == "__main__" :

    # Select what conditions to compute
    cond1 = "LaughReal"
    cond2 = "LaughPosed"
    picks = "meg" # Select MEG channels
    event_id = {'LaughReal' : 11, 'LaughPosed' : 12}

    # TODO : Need to put that in params
    save_cond1 = RESULT_PATH + "meg/reports/epochs/ave-epo_cond-{}_epochs.pkl".format(cond1)
    save_cond2 = RESULT_PATH + "meg/reports/epochs/ave-epo_cond-{}_epochs.pkl".format(cond2)
    save_concat = RESULT_PATH + "meg/reports/epochs/concat-epo_cond-{}-{}_epochs.pkl".format(cond1, cond2)

    # Need to get the files
    with open(save_cond1, "rb") as f:
        condition1 = pickle.load(f)

    with open(save_cond2, "rb") as f:
        condition2 = pickle.load(f)

    with open(save_concat, "rb") as f:
        epochs_concat = pickle.load(f)

    # Plot ERPs
    plot_ERP(condition1, condition2, cond1, cond2, picks)

    # TODO : Need to find a better way to import these files
    alpha_cluster_forming = 0.001
    save_Fobs = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-Fobs_pval-{}.pkl".format(alpha_cluster_forming)
    save_clusters = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-clusters_pval-{}.pkl".format(alpha_cluster_forming)
    save_pval = RESULT_PATH + "meg/reports/epochs/ERP-cluster_mes-pva_pval-{}.pkl".format(alpha_cluster_forming)

    with open(save_Fobs, 'rb') as f:
        F_obs = pickle.load(f)

    with open(save_clusters, 'rb') as f:
        clusters = pickle.load(f)
    
    with open(save_pval, 'rb') as f:
        p_values = pickle.load(f)

    # Visualization of ERP clusters
    visualize_cluster(epochs_concat, F_obs, clusters, p_values, event_id)
