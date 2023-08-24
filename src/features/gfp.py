
import mne
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, ACTIVE_RUN, RESULT_PATH, EVENTS_ID, SUBJ_CLEAN, FIG_PATH

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
args = parser.parse_args()


def compute_gfp(task, cond1, cond2) :

    evo_grand_average_cond1 = mne.read_evokeds(RESULT_PATH + 'meg/reports/sub-all/erp/sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.fif'.format(task, cond1))
    evo_grand_average_cond2 = mne.read_evokeds(RESULT_PATH + 'meg/reports/sub-all/erp/sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.fif'.format(task, cond2))

    evo_grand_average_cond1 = evo_grand_average_cond1[0]
    evo_grand_average_cond2 = evo_grand_average_cond2[0]

    # Plot gfp for one condition 
    evo_grand_average_cond1.plot(gfp='only')
    plt.savefig(FIG_PATH + 'gfp/sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.png'.format(task, cond1))

    evo_grand_average_cond2.plot(gfp='only')
    plt.savefig(FIG_PATH + 'gfp/sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.png'.format(task, cond2))


    # Plot gfp of two conditions
    gfp_cond1 = evo_grand_average_cond1[0].data.std(axis=0, ddof=0)
    gfp_cond2 = evo_grand_average_cond2[0].data.std(axis=0, ddof=0)

    # Reproducing the MNE-Python plot style seen above
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    ax.plot(evo_grand_average_cond1[0].times, gfp_cond1 * 1e15, label = cond1, c='g')
    ax.plot(evo_grand_average_cond2[0].times, gfp_cond2 * 1e15, label = cond2, c='r')
    ax.legend()

    #ax.fill_between(evo_grand_average.times, gfp * 1e15, alpha=0.2)
    ax.set(xlabel="Time (s)", ylabel="GFP (µV)", title="MEG")

    plt.savefig(FIG_PATH + 'gfp/sub-all_task-{}_run-all_cond-{}-{}_meas-grandave-ave.png'.format(task, cond1, cond2))


if __name__ == "__main__" :

    cond1 = args.condition1
    cond2 = args.condition2

    task = args.task

    compute_gfp(task, cond1, cond2)