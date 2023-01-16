import mne
import os
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, FIG_PATH
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mne.stats import spatio_temporal_cluster_test, combine_adjacency
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.viz import plot_compare_evokeds

# Will be moved in another file later
def ERP(PREPROC_PATH, subj_list, run_list, cond1, cond2, stage) :
    for subj in subj_list :
        for run in run_list :
            epochs_file = get_bids_file(PREPROC_PATH, subj, run,  stage)
            epochs = mne.read_epochs(epochs_file)
            print(epochs.info)

            # Average each condition
            condition1 = epochs[cond1].average()
            condition2 = epochs[cond2].average()
    return condition1, condition2

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
    fig_ERP = mne.viz.plot_compare_evokeds(evoked, picks = picks, combine='mean')
    fname_ERP = FIG_PATH + "ERP_cond1-" + cond1_name + "_cond2-" + cond2_name
    # fig_ERP.savefig(fname_ERP) # Doesn't work

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
    cluster_stats = spatio_temporal_cluster_test(X, n_permutations=50,
                                                threshold=f_thresh, tail=tail,
                                                n_jobs=None, buffer_size=None,
                                                adjacency=None)
    F_obs, clusters, p_values, _ = cluster_stats

    return F_obs, clusters, p_values

def visualize_cluster(epochs, F_obs, clusters, p_values, event_id) :

    # Code adapted from :
    # https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html
    
    p_accept = 0.5
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
        "Done Visualize cluster"
        return fig

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    subj_list = ["02"]
    run_list = ["07"]
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = "LaughReal"
    cond2 = "LaughPosed"
    picks = "meg" # Select MEG channels
    event_id = {'LaughReal' : 11, 'LaughPosed' : 12}
    # Compute ERPs
    #condition1, condition2 = ERP(PREPROC_PATH, subj_list, run_list, cond1, cond2, "epo")

    # Plot ERPs
    #plot_ERP(condition1, condition2, cond1, cond2, picks)

    # Compute ERP clusters
    # For now N = 1, need to check with more participants    
    for subj in subj_list :
        for run in run_list :
            epochs_file = get_bids_file(PREPROC_PATH, subj, run,  stage)
            epochs = mne.read_epochs(epochs_file)

    F_obs, clusters, p_values = cluster_ERP(epochs, event_id)

    # Visualization of ERP clusters
    fig = visualize_cluster(epochs, F_obs, clusters, p_values, event_id)
