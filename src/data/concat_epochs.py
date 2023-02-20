import mne
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_LIST, ACTIVE_RUN, RESULT_PATH, EVENTS_ID

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

# Will be moved in another file later
def get_epochs(PREPROC_PATH, subj_list, event_id, task, cond1, cond2, stage) :
    
    epochs_concat = []
    epochs_all_list = []
    epo_condition1 = []
    epo_condition2 = []

    for subj in subj_list :
        print('processing -->', subj)

        # Import epochs file
        _, epochs_file = get_bids_file(PREPROC_PATH, task = task, stage = stage, subj = subj)
        epochs = mne.read_epochs(epochs_file)
        epochs = mne.Epochs.apply_baseline(epochs,baseline=(None,0))
        epochs.equalize_event_counts(event_id) # Equalize events number

        # Set manually head mouvement 
        if subj == subj_list[0]:
            head_info = epochs.info['dev_head_t']
        else:
            epochs.info['dev_head_t'] = head_info
        
        # Concatenate epochs regardless conditions
        epochs_concat.append(mne.concatenate_epochs([epochs]))

        # Concatenate each condition separately
        epo_condition1.append(mne.concatenate_epochs([epochs[cond1]]))
        epo_condition2.append(mne.concatenate_epochs([epochs[cond2]]))
        del epochs

    # Create a dict to optimize save files
    conditions = {cond1 : epo_condition1, cond2: epo_condition2}

    # Save average condition into pkl file
    for one_condition in conditions :
        _, save_erp = get_bids_file(RESULT_PATH, task=task, stage = "erp", condition=one_condition)
        with open(save_erp, 'wb') as f:
            pickle.dump(conditions.get(one_condition), f)
    
    # Save concat condition into pkl file
    conditions = cond1 + "-" + cond2
    _, save_erp_concat = get_bids_file(RESULT_PATH, task=task, stage = "erp-concat", condition=conditions)
    
    with open(save_erp_concat, 'wb') as f:
        pickle.dump(epochs_concat, f)    

    return epo_condition1, epo_condition2, epochs_concat

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = ["01", '02']
    stage = "epo"

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
    conditions = conditions = cond1 + '-' + cond2
    condition_list = [cond1, cond2]
    event_id = dict()
    picks = "meg" # Select MEG channels
    
    for ev in EVENTS_ID :
        for conds in condition_list :
            if conds not in EVENTS_ID :
                raise Exception("Condition is not an event")
            if conds == ev :
                event_id[conds] = EVENTS_ID[ev] # Select event ID of interest

    print("=> Process task :", task, "for conditions :", cond1, "&", cond2)

    # Compute epochs for each condition
    epo_condition1, epo_condition2, epochs_concat = get_epochs(PREPROC_PATH, subj_list, event_id, task, cond1, cond2, "proc-clean_epo")