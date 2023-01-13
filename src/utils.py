import os
from src.params import ACTIVE_RUN, PASSIVE_RUN, PREPROC_PATH

def get_bids_file(BIDS_PATH, subj, run, stage) :
    # Step 1 : get all parameters of the file name : subj, session, task
    # Step 2 : stage gives us the extension and at what step in the processing we are => extension file
    # Step 3 : put all the information in var laughter_bidsname
    # Step 4 : Return laughter_bidspath

    # stages : ave, epo, ica, cov, ica_epo, filt_raw, clean epo => .fif
    # other stages : report (html)

    # Find active and passive runs
    if run in ACTIVE_RUN :
        task = "LaughterActive"
    elif run in PASSIVE_RUN :
        task = "LaughterPassive"

    # Raw data 
    if  stage == "raw":
        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_meg.ds".format(subj, task, run)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)

    # Preprocessed data 
    elif (stage == "ave"  
    or stage == "epo" 
    or stage == "ica"
    or stage == "cov"  
    or stage == "ica_epo"
    or stage == "clean epo") : 
        
        extension = ".fif"

        laughter_bidsname = "sub-{}_ses-recording_task-{}_{}{}".format(subj, task, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)
    
    elif stage == "filt_raw" :
        extension = ".fif"
        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}{}".format(subj, task, run, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)
    
    # Analysed data

    return laughter_bidspath
