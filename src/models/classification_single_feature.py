import mne 
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, PREPROC_PATH, FREQS_LIST
from src.utils import get_bids_file
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
)

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
    help="Condition to compute",
)
parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-freq",
    "--freq",
    default="alpha",
    type=str,
    help="Frquency to compute",
)
parser.add_argument(
    "-tmin",
    "--tmin",
    default=0.0,
    type=float,
    help="Tmin to compute",
)
parser.add_argument(
    "-tmax",
    "--tmax",
    default=0.5,
    type=float,
    help="Tmin to compute",
)
args = parser.parse_args()

def prepare_data(subj_list, run_list, freq_name, cond1, cond2, classifier, clf_name) : 

    epochs_list = []

    y_run_cond1 = []
    y_run_cond2 = []

    X_subj = []
    y_subj = []

    for subj in ['01']: 

        print(subj)
        
        for run in ['07'] :
            
            # importer les psd (enveloppe spectrale)
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, run=run, measure=freq_name, stage="psd_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            
            epochs.apply_baseline(baseline=(None, 0))
            epochs.resample(50)

            if subj == subj_list[0] and run == run_list[0]:
                head_info = epochs.info['dev_head_t']
            else:
                epochs.info['dev_head_t'] = head_info
                
            epochs_list.append(epochs)
            
            del epochs
            
        epochs_all_run = mne.concatenate_epochs(epochs_list)  
        
        # Compute power (= envelop **2)
        power_cond1 = epochs_all_run[cond1].get_data()**2 # Shape n_epochs, n_chan, n_times
        power_cond2 = epochs_all_run[cond2].get_data()**2 # Shape n_epochs, n_chan, n_times

        X_subj.append(np.concatenate((power_cond1, power_cond2)))
        y_subj.append(np.concatenate((epochs_all_run[cond1].events[:, 2], epochs_all_run[cond2].events[:, 2])))

    X = X_subj[-1] #take the last concatenation
    y = y_subj[-1]

    print('Shape X', X.shape)
    print('Shape y', y.shape)

    decoding_time(X, y, classifier, cond1, cond2, clf_name, epochs=epochs_all_run, sfs=True)

    return X, y 

def decoding_time(X, y, classifier, cond1, cond2, clf_name, epochs, sfs=True) : 

    conditions = cond1 + '-' + cond2

    # TODO : put savepoint 
    # We will train the classifier on all left visual vs auditory trials on MEG
    sfs = SequentialFeatureSelector(classifier, n_features_to_select=5)

    clf = make_pipeline(StandardScaler(), sfs, classifier)

    time_decod = SlidingEstimator(clf, scoring="roc_auc", verbose=True, n_jobs=-1)
    # here we use cv=3 just for speed
    scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=-1)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)

    # Plot
    fig, ax = plt.subplots()

    ax.plot(epochs.times, scores, label="score")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Times")
    ax.set_ylabel("AUC")  # Area Under the Curve
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title("Sensor space decoding")

    plt.savefig(FIG_PATH + 'ml/sub-01_task-{}_run-all_cond-{}_meas-decoding-time_{}-5_features-.png'.format(task, conditions, clf_name))


def time_window_hilbert(subj_list, RUN_LIST, tmin, tmax, conditions, FREQS_NAMES, FREQS_LIST) :
    
    print('Time window : ' + str(tmin) + '-' + str(tmax))
    epochs_list = []

    y_run_cond1 = []
    y_run_cond2 = []

    X_subj = []
    y_subj = []
    
    power_list = []
    FREQS = [x for x in range(len(FREQS_LIST))]
    idx = 0 #TODO : Change that
    
    group = []

    for l_freq, h_freq in FREQS_LIST :
        print('Processing freqs -->', l_freq, h_freq)
        
        for i_subj, subj in enumerate(subj_list):
            print("=> Process task :", task, 'subject', subj)

            sub_id = 'sub-' + subj
            subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id)

            if not os.path.isdir(subj_path):
                os.mkdir(subj_path)
                print('BEHAV folder created at : {}'.format(subj_path))
            else:
                print('{} already exists.'.format(subj_path))

            # Take ica data
            _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
            ica = mne.preprocessing.read_ica(ica_path)
        
            # Select rejection threshold associated with each run
            # TODO : apply threshold per run
            print('Load reject threshold')
            _, reject_path = get_bids_file(RESULT_PATH, stage="AR_epo", task=task, subj=subj, measure='log')

            with open(reject_path, 'rb') as f:
                reject = pickle.load(f)  

            # Select event_id appropriate
            for run in RUN_LIST :
                print("=> Process run :", run)
                        
                if task == 'LaughterPassive' :
                    if run == '02' or run == '03' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 
                                    'EnvPosed' : 22}
                    elif run == '04' or run == '05' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12,'ScraReal' : 31, 
                                    'ScraPosed' : 32,}
                if task == 'LaughterActive' :
                        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66}


                # Take raw data filtered : i.e. NO ICA 
                print('ICA')
                _, raw_path = get_bids_file(PREPROC_PATH, stage='proc-filt_raw', task=task, run=run, subj=subj)
                raw = mne.io.read_raw_fif(raw_path, preload=True)
                raw_filter = raw.copy()
                raw_filter = ica.apply(raw_filter)

                epochs_psds = []

                freq_name = FREQS_NAMES[idx]

                print('Apply filter')
                info = raw_filter.info
                raw_filter.filter(l_freq, h_freq) # Apply filter of interest
                raw_hilbert = raw_filter.apply_hilbert(envelope=True)

                picks = mne.pick_types(raw.info, meg=True, ref_meg=False, eeg=False, eog=False)

                power_hilbert = raw_hilbert.copy()
                power_hilbert._data = power_hilbert._data**2

                # Segmentation
                print('Segmentation')
                events = mne.find_events(raw)
                epochs = mne.Epochs(
                    power_hilbert,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=(0, 0),
                    picks=picks)    

                # Auto-reject epochs
                epochs.drop_bad(reject=reject)

                if subj == subj_list[0] and run == run_list[0]:
                    head_info = epochs.info['dev_head_t']
                else:
                    epochs.info['dev_head_t'] = head_info

                epochs.equalize_event_counts(['LaughReal', 'LaughPosed'])
                epochs_list.append(epochs)

                del epochs
                
        epochs_all_run = mne.concatenate_epochs(epochs_list)  

        print(epochs_all_run)
        
        # Compute power (= envelop **2)
        print('Calculte power')
        ave_time_cond1 = np.mean(epochs_all_run[cond1].get_data(), axis=2)
        ave_time_cond2 = np.mean(epochs_all_run[cond2].get_data(), axis=2)
        
        print('Concatenation')
        X_subj.append(np.concatenate((ave_time_cond1, ave_time_cond2)))
        y_subj.append(np.concatenate((epochs_all_run[cond1].events[:, 2], epochs_all_run[cond2].events[:, 2])))

        for i_cond, cond in enumerate(X_subj[-1]) : 
            group.append(i_subj)
        
        print(X_subj[-1].shape)
        print(y_subj[-1].shape)
        print(np.array(group).shape)
        
        sub_name = 'all'
        run_name = 'all'
        tmin_name = str(int(tmin*1000))
        tmax_name = str(int(tmax*1000))
        stage = 'ml-'+ tmin_name + '-' + tmax_name

        save_X = RESULT_PATH + 'meg/reports/sub-all/ml/sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}.pkl'.format(sub_name, task, 
                                                                                    run_name, conditions,
                                                                                    'X-subj', stage)
        
        save_y = RESULT_PATH + 'meg/reports/sub-all/ml/sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}.pkl'.format(sub_name, task, 
                                                                                    run_name, conditions,
                                                                                    'y-subj', stage)
        save_group = RESULT_PATH + 'meg/reports/sub-all/ml/sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}.pkl'.format(sub_name, task, 
                                                                                    run_name, conditions,
                                                                                    'group', stage)
        with open(save_X, 'wb') as f:
            pickle.dump(X_subj, f)
    
        with open(save_y, 'wb') as f:
            pickle.dump(y_subj, f)

        with open(save_group, 'wb') as f:
            pickle.dump(group, f)


        arr_scores = classif_single_chan(X_subj, y_subj, group) 

    return X_subj, y_subj

def classif_single_chan(X_subj, y_subj, group) : 

    CHAN = np.arange(0, 270, 1)

    scores_chan = []

    # Select the classifier & cross-val method
    clf = svm.SVC(kernel='linear', C=1, random_state=42)

    cv = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        
    X = X_subj[-1] #X_subj[-1]
    print(X.shape)

    y = y_subj[-1]
    print(y.shape)

    groups = np.array(group)

    X_random = np.random.rand(3240, 10)


    for chan in CHAN :
        print('Process channel :', chan)

        # Select channel of interest
        X_chan = X[:, chan]
        X_classif = X_chan.reshape(-1, 1) # Reshape (n_event, 1)
        print(X_chan.shape)
        print(X_classif.shape)
        
        scores = cross_val_score(clf, X=X_classif, y=y, groups=groups, cv=cv)
        print(scores)
        
        # Mean scores across cross-validation splits
        scores_mean = np.mean(scores, axis=0)
        print('Score :', scores_mean)

        # Put results into list
        scores_chan.append(scores_mean)

    # Put all scores into numpy
    arr_scores = np.array(scores_chan)
    

    return arr_scores
    
if __name__ == "__main__" :

    subj_list = SUBJ_CLEAN
    task = args.task
    cond1 = args.condition1
    cond2 = args.condition2
    freq_name = args.freq
    conditions = cond1 + '-' + cond2

    tmin = args.tmin
    tmax = args.tmax

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    # Select what conditions to compute
    condition_list = [cond1, cond2]

    classifier = LogisticRegression(solver="liblinear")
    clf_name = 'LR'

    hilbert_window = True
    time_decoding = False

    if time_decoding == True : 
        X, y = prepare_data(subj_list, run_list, 
                            freq_name, 
                            cond1, cond2, 
                            classifier, clf_name)

    elif hilbert_window == True : 
        X_subj, y_subj, group, epochs_all_run = time_window_hilbert(subj_list, run_list, 
                                                                    tmin, tmax, 
                                                                    cond1, cond2,
                                                                    FREQS_NAMES=[freq_name], 
                                                                    FREQS_LIST=[FREQS_LIST[2]], 
                                                                    save=True)


    