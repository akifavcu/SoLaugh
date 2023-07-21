import pickle
import os
import mne
import argparse
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from src.params import PREPROC_PATH, SUBJ_CLEAN, RESULT_PATH, RUN_LIST, PASSIVE_RUN, ACTIVE_RUN, RS_RUN
from src.utils import get_bids_file

# See https://autoreject.github.io/stable/index.html

parser = argparse.ArgumentParser()
parser.add_argument(
    "-subj",
    "--subject",
    default="01",
    type=str,
    help="Subject to process",
)
args = parser.parse_args()

subject = args.subject
SUBJ_LIST = [subject]

task_list = ['LaughterActive', 'LaughterPassive']
# Import epochs preprocessed data
for task in task_list : 

    for subj in SUBJ_LIST : 

        # Prepare path
        sub_id = 'sub-' + subj
        subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id)
        print(sub_id)

        if not os.path.isdir(subj_path):
            os.mkdir(subj_path)
            print('BEHAV folder created at : {}'.format(subj_path))
        else:
            print('{} already exists.'.format(subj_path))

        # First reject automatically epochs
        _, epoch_path = get_bids_file(PREPROC_PATH, stage="proc-clean_epo", task=task, subj=subj)
        epochs = mne.read_epochs(epoch_path)
        epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)

        # Export in dict epochs rejected
        reject = get_rejection_threshold(epochs)  
        print('Autoreject: Done')  

        # Save autoreject epochs
        _, save_AR_epochs = get_bids_file(PREPROC_PATH, stage="AR_epo", task=task, subj=subj)
        epochs_clean.save(save_AR_epochs, overwrite = True) 

        # Save dict with rejected epochs info
        _, save_AR = get_bids_file(PREPROC_PATH, stage="AR_epo", task=task, subj=subj, measure='log')

        with open(save_AR, 'wb') as f:
            pickle.dump(reject, f)

        print('Save: Done')
