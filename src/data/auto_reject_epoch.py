import pickle
import os
import mne
import argparse
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from src.params import PREPROC_PATH, SUBJ_CLEAN, RESULT_PATH, RUN_LIST, PASSIVE_RUN, ACTIVE_RUN, RS_RUN, EVENTS_ID
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

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive' :
        run_list = PASSIVE_RUN

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

        _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
        ica = mne.preprocessing.read_ica(ica_path)
         
        for run in run_list : 

            # Select filtered raw files
            _, raw_path = get_bids_file(PREPROC_PATH, stage="proc-filt_raw", run=run, task=task, subj=subj)
            raw = mne.io.read_raw_fif(raw_path, preload=True)
            raw.pick_types(meg=True, ref_meg=False,  exclude='bads')

            # Apply ICA
            raw_filter = raw.copy()
            raw_filter = ica.apply(raw_filter)
            
            # Create epochs
            info = raw_filter.info
            picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)

            '''
            if run == '02' or run == '03' : 
                event_id = {'Laugh/Real' : 11, 'Laugh/Posed' : 12, 'Env/Real' : 21, 
                    'Env/Posed' : 22}
            elif run == '04' or run == '05' :
                event_id = {'Laugh/Real' : 11, 'Laugh/Posed' : 12, 'Scra/Real' : 31, 
                    'Scra/Posed' : 32}
            else : 
                event_id = {'Laugh/Real' : 11, 'Laugh/Posed' : 12, 'Press/Good' : 99, 
                    'Press/Miss' : 66} '''

            # Segmentation
            events = mne.find_events(raw)
            epochs = mne.Epochs(
                raw_filter,
                events=events,
                event_id=EVENTS_ID,
                tmin=-0.5,
                tmax=1.5,
                baseline=(None, 0),
                picks=picks)
            
            # Apply AutoReject
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epochs)

            # Export in dict epochs rejected
            reject = get_rejection_threshold(epochs)  
            print('Autoreject: Done')  

            # Save autoreject epochs
            _, save_AR_epochs = get_bids_file(RESULT_PATH, stage="AR_epo", task=task, subj=subj, run=run)
            epochs_clean.save(save_AR_epochs, overwrite = True) 

            # Save dict with rejected epochs info
            _, save_AR = get_bids_file(RESULT_PATH, stage="AR_epo", task=task, subj=subj, measure='log', run=run)

            with open(save_AR, 'wb') as f:
                pickle.dump(reject, f)

            print('Save: Done')
