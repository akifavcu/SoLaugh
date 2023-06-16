import pickle
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from src.params import PREPROC_PATH, SUBJ_CLEAN
from src.utils import get_bids_file

# See https://autoreject.github.io/stable/index.html

# Import epochs preprocessed data
for subj in SUBJ_CLEAN : 
   
    # First reject automatically epochs
    epochs = get_bids_file(PREPROC_PATH, stage="proc-clean_epo", task=task, subj=subj)
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)  

    # Export in dict epochs rejected
    reject = get_rejection_threshold(epochs)  

    # Save autoreject epochs
    _, save_AR_epochs = get_bids_file(RESULT_PATH, stage = "AR", task=task, subj=subj, measure='epo')
    epochs_clean.save(save_AR_epochs, overwrite = True) 

    # Save dict with rejected epochs info
    _, save_AR = get_bids_file(RESULT_PATH, stage = "AR", task=task, subj=subj, measure='log')

    with open(save_AR, 'wb') as f:
        pickle.dump(reject, f)  
