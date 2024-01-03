import mne 
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, PREPROC_PATH, FREQS_LIST, FREQS_NAMES
from src.utils import get_bids_file
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, RandomizedSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm


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

parser.add_argument(
    "-compute_hilbert",
    "--compute_hilbert",
    default=True,
    type=bool,
    help="Wwither or not computing Hilbert",
)
args = parser.parse_args()

def save_data(subj, save, tmin, tmax, cond1, cond2, X_subj, y_subj, X_cond1, X_cond2, group, freq_name) : 
    sub_name = subj
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = 'ml-'+ tmin_name + '-' + tmax_name
    conditions = cond1 + '-' + cond2

    save_X = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-subj', freq_name, stage)
    save_y = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'y-subj', freq_name, stage)

    save_group = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'group', freq_name, stage)

    save_X1 = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond1', freq_name, stage)

    save_X2 = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond2', freq_name, stage)
    with open(save_X, 'wb') as f:
        pickle.dump(X_subj, f)

    with open(save_y, 'wb') as f:
        pickle.dump(y_subj, f)

    with open(save_group, 'wb') as f:
        pickle.dump(group, f)

    with open(save_X1, 'wb') as f:
        pickle.dump(X_cond1, f)

    with open(save_X2, 'wb') as f:
        pickle.dump(X_cond2, f)

def upload_data(subj_list, tmin, tmax, cond1, cond2, freq_name) : 
    for subj in subj_list : 
        sub_name = subj
        run_name = 'all'
        tmin_name = str(int(tmin*1000))
        tmax_name = str(int(tmax*1000))
        stage = tmin_name + '-' + tmax_name
        conditions = cond1 + '-' + cond2

        save_X = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'X-subj', freq_name, stage)
        
        save_y = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'y-subj', freq_name, stage)

        save_group = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                                run_name, conditions,
                                                                                                                'group', freq_name, stage)

        save_X1 = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'X-cond1', freq_name, stage)

        save_X2 = RESULT_PATH + 'meg/reports/sub-{}/ml/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq-{}_{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'X-cond2', freq_name, stage)

        with open(save_X, 'rb') as f:
            X_subj = pickle.load(f)
            
        with open(save_y, 'rb') as f:
            y_subj = pickle.load(f)
            
        with open(save_group, 'rb') as f:
            group = pickle.load(f)
            
        with open(save_X1, 'rb') as f:
            X_cond1 = pickle.load(f)

        with open(save_X2, 'rb') as f:
            X_cond2 = pickle.load(f)

        arr_scores = classif_single_chan(subj, X_subj, y_subj, group, tmin, tmax)

    return X_subj, y_subj, X_cond1, X_cond2

def time_window_hilbert(subj_list, RUN_LIST, tmin, tmax, cond1, cond2, FREQS_NAMES, FREQS_LIST, save=True) :
    
    print('Time window : ' + str(tmin) + '-' + str(tmax))

    
    FREQS = [x for x in range(len(FREQS_LIST))]
    idx = 0 


    for l_freq, h_freq in FREQS_LIST :
        print('Processing freqs -->', l_freq, h_freq)
        
        for i_subj, subj in enumerate(subj_list):
            print("=> Process task :", task, 'subject', subj)
            epochs_list = []

            X_subj = []
            y_subj = []

            X_cond1 = []
            X_cond2 = []

            group = []
            
            sub_id = 'sub-' + subj
            subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id, 'ml', 'results_single_feat')

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

                freq_name = FREQS_NAMES[idx]

                print('Apply filter')
                info = raw_filter.info
                raw_filter.filter(l_freq, h_freq, n_jobs=-1) # Apply filter of interest
                raw_hilbert = raw_filter.apply_hilbert(envelope=True, n_jobs=-1)

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
                    baseline=(tmin, tmin),
                    picks=picks)    

                # Auto-reject epochs
                epochs.drop_bad(reject=reject)

                if subj == subj_list[0] and run == run_list[0]:
                    head_info = epochs.info['dev_head_t']
                else:
                    epochs.info['dev_head_t'] = head_info

                epochs.equalize_event_counts([cond1, cond2]) # Equalize event count
                epochs_list.append(epochs)

                for i_cond, cond in enumerate(epochs[cond1, cond2].get_data()) : 
                    group.append(i_subj)

                del epochs
                
            epochs_all_run = mne.concatenate_epochs(epochs_list)  

            print(epochs_all_run)
            
            # Compute power (= envelop **2)
            print('Calculte power')
            ave_time_cond1 = np.mean(epochs_all_run[cond1].get_data(), axis=2)
            ave_time_cond2 = np.mean(epochs_all_run[cond2].get_data(), axis=2)
            
            X_cond1.append(ave_time_cond1)
            X_cond2.append(ave_time_cond2)
            
            print('Concatenation')
            X_subj.append(np.concatenate((ave_time_cond1, ave_time_cond2)))
            y_subj.append(np.concatenate((epochs_all_run[cond1].events[:, 2], epochs_all_run[cond2].events[:, 2])))
            
            print(X_subj[-1].shape)
            print(y_subj[-1].shape)
            print(np.array(group).shape)
            

            if save == True : 
                save_data(subj, save, tmin, tmax, cond1, cond2, X_subj, y_subj, X_cond1, X_cond2, group, freq_name)

            arr_scores = classif_single_chan(subj, X_subj, y_subj, group, tmin, tmax, freq_name) 

    return X_subj, y_subj, group

def classif_single_chan(subj, X_subj, y_subj, group, tmin, tmax, freq_name) : 

    CHAN = np.arange(0, 270, 1)

    all_scores = []

    X = X_subj[-1] #X_subj[-1]
    print(X.shape)

    y = y_subj[-1]
    print(y.shape)

    groups = np.array(group)
    print(group)

    for chan in CHAN :
        print('-->Process channel :', chan)
        
        y_pred = []
        
        parameters = {'kernel':('linear', 'rbf',  'rbf', 'sigmoid'), 'C':[1, 10]}
        clf = svm.SVC()
        #parameters = {'loss' : ('log_loss', 'exponential'), 
        #              'criterion' : ('friedman_mse', 'squared_error')}
        #clf = GradientBoostingClassifier()
        
        cv = LeaveOneOut() # Cross-validation via LOGO
        
        # Select channel of interest
        X_chan = X[:, chan]
        X_classif = X_chan.reshape(-1, 1) # Reshape (n_event, 1)
        print(X_classif.shape)
        
        # Find best params with GridSearch
        # Use RandomSearch
        print('---->Find best params')
        search = RandomizedSearchCV(clf, parameters, cv=cv, verbose=True, n_jobs=-1).fit(X=X_classif, 
                                                                                        y=y, 
                                                                                        groups=groups)
            
        # Select best params
        best_params = search.best_estimator_
        print('Best params : ' + str(best_params))

        # Accuracy score
        scores = search.score(X=X_classif, y=y) 
        print('Scores', scores)
        #print('Best scores', scores.best_score_)
        
        all_scores.append(scores)

    sub_name = subj
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = tmin_name + '-' + tmax_name 
    conditions = cond1 + '-' + cond2
    score = 'scores-svm'

    save_scores = RESULT_PATH + 'meg/reports/sub-{}/ml/results_single_feat/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_freq_{}{}.pkl'.format(sub_name, sub_name, task, 
                                                                                                                                run_name, conditions,
                                                                                                                                score, freq_name, stage)
    with open(save_scores, 'wb') as f:
        pickle.dump(all_scores, f)
    
    return all_scores
    
if __name__ == "__main__" :

    subj_list = SUBJ_CLEAN
    task = args.task

    # Condition 1 and 2
    cond1 = args.condition1
    cond2 = args.condition2
    
    # Frequence selected
    freq_name = args.freq    
    
    # Time selection
    tmin = args.tmin
    tmax = args.tmax

    # Weither or not to compute hilbert or upload data
    compute_hilbert = args.compute_hilbert
    print(compute_hilbert)

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    # Select what conditions to compute
    condition_list = [cond1, cond2]
    
    frequency_list = []
    for i, freq in enumerate(FREQS_NAMES) : 
        if freq == freq_name :
            frequency_list.append(FREQS_LIST[i])

    if compute_hilbert == True : 
        X_subj, y_subj, group = time_window_hilbert(subj_list, run_list, 
                                                                    tmin, tmax, 
                                                                    cond1, cond2,
                                                                    FREQS_NAMES=[freq_name], 
                                                                    FREQS_LIST=frequency_list, 
                                                                    save=True)
    else :
        upload_data(subj_list, tmin, tmax, cond1, cond2, freq_name)




    