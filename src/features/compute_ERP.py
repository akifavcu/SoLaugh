import mne
import os
import matplotlib.pyplot as plt
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, FIG_PATH

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
    evoked.plot_joint(picks = picks)

    fname_ERP = FIG_PATH + "ERP_cond1-" + cond1_name + "_cond2-" + cond2_name
    # fig_ERP.savefig(fname_ERP) # Doesn't work

# def cluster_ERP
# See https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    subj_list = ["02"]
    run_list = ["07"]
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = "LaughReal"
    cond2 = "LaughPosed"
    picks = "meg" # Select MEG channels

    # Compute ERPs
    condition1, condition2 = ERP(PREPROC_PATH, subj_list, run_list, cond1, cond2, "epo")

    # Plot ERPs
    plot_ERP(condition1, condition2, cond1, cond2, picks)
