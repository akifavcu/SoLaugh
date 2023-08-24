import mne
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, RESULT_PATH, EVENTS_ID, FIG_PATH

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-cond",
    "--condition",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
args = parser.parse_args()


def grand_ave_erp(task, cond) :
    
    evoked_one_condition = []

    for subj in SUBJ_CLEAN :

        print("processing -->", subj)

        # TODO : change with AR_epochs
        _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
        epochs = mne.read_epochs(path_epochs, verbose=None)
        
        # Drop EEg channels and equalize event number
        evoked_one_condition.append(epochs[cond].average()) 
        
    evokeds_all_subj = mne.combine_evoked(evoked_one_condition, 'equal')

    evokeds_all_subj.save(RESULT_PATH + 'meg/reports/sub-all/erp/sub-all_task-{}_run-all_cond-{}_meas-grandave-ave.fif'.format(task, cond))

    return evokeds_all_subj

if __name__ == "__main__" :

    cond = args.condition
    task = args.task

    grand_average_erp = grand_ave_erp(task, cond)