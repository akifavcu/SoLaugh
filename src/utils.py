import os
from src.params import ACTIVE_RUN, PASSIVE_RUN, PREPROC_PATH

def get_bids_file(BIDS_PATH, stage, subj='all', run='all', task = "LaughterActive", measure=None, condition=None) :
    
    """
    conditions = list()
    TODO : check task param
    If task is set manually, and the run doesn't fit,
    task will be automatically change
    If run = 'all', by default task is LaughterActive
    """
    print(stage)
    # Find active and passive runs
    if run in ACTIVE_RUN :
        task = "LaughterActive"
    elif run in PASSIVE_RUN :
        task = "LaughterPassive"

    # Take raw data 
    if  stage == "raw":
        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_meg.ds".format(subj, task, run)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)

    # Take preprocessed data 
    elif (stage == "ave"  
    or stage == "epo" 
    or stage == "ica"
    or stage == "cov"  
    or stage == "ica_epo"
    or stage == "clean epo") : 
        print('in epo')
        extension = ".fif"

        laughter_bidsname = "sub-{}_ses-recording_task-{}_{}{}".format(subj, task, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)
    
    elif stage == "filt_raw" :

        extension = ".fif"

        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}{}".format(subj, task, run, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), "ses-recording", "meg", laughter_bidsname)

    # ERPs and PSD files
    
    elif ("psd" in stage or "erp" in stage) :

        folder = stage[0:3]
        extension = ".pkl"

        if condition != None :
            if measure == None :
                laughter_bidsname = "sub-{}_run-{}_task-{}_{}_cond-{}{}".format(subj, run, task, 
                                                                                stage, condition, extension)

                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", "epochs", folder, laughter_bidsname)
            else :
                laughter_bidsname = "sub-{}_run-{}_task-{}_{}_cond-{}_meas-{}{}".format(subj, run, 
                                                                                        task, stage, condition,
                                                                                        measure, extension)
                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", "epochs", folder, laughter_bidsname)
        else :
            raise Exception("Missing argument : conditions")

    return laughter_bidsname, laughter_bidspath
