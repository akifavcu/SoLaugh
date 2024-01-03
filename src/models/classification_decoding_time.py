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
from sklearn.model_selection import cross_val_score, ShuffleSplit, GroupShuffleSplit, LeaveOneGroupOut
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
    # sfs = SequentialFeatureSelector(classifier, n_features_to_select=5)
    cv = LeaveOneGroupOut() # Cross-validation via LOGO
    clf = make_pipeline(cv, classifier)

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

    plt.savefig(FIG_PATH + 'ml/sub-01_task-{}_run-all_cond-{}_meas-decoding-time_{}.png'.format(task, conditions, clf_name))

if __name__ == "__main__" :

    subj_list = SUBJ_CLEAN
    task = args.task
    cond1 = args.condition1
    cond2 = args.condition2
    freq_name = args.freq

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


    X, y = prepare_data(subj_list, run_list, 
                            freq_name, 
                            cond1, cond2, 
                            classifier, clf_name)