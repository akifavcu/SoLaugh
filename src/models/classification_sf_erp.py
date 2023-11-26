# Utils
from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN
from src.utils import get_bids_file
import os
import pickle
import argparse


import numpy as np
import mne 

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut


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
    "-tmin",
    "--tmin",
    default=0.7,
    type=float,
    help="Tmin to compute",
)
parser.add_argument(
    "-tmax",
    "--tmax",
    default=0.750,
    type=float,
    help="Tmin to compute",
)

parser.add_argument(
    "-compute_erp",
    "--compute_erp",
    default=True,
    type=bool,
    help="Weither or not processing erp",
)
args = parser.parse_args()

def save_data(save, tmin, tmax, cond1, cond2, X_subj, y_subj, X_cond1, X_cond2, group) : 
    sub_name = 'all'
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = 'ml-erp'+ tmin_name + '-' + tmax_name
    conditions = cond1 + '-' + cond2

    save_X = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-subj', stage)
    save_y = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'y-subj', stage)

    save_group = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'group', stage)

    save_X1 = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond1', stage)

    save_X2 = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond2', stage)
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

def upload_data(tmin, tmax, cond1, cond2) : 
    sub_name = 'all'
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = 'ml-erp'+ tmin_name + '-' + tmax_name
    conditions = cond1 + '-' + cond2

    save_X = os.path.join(RESULT_PATH, 'meg/reports',
                          'sub-all', 'ml/erp',
                          'sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                    run_name, conditions,
                                                                                    'X-subj', stage))
    
    save_y = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'y-subj', stage)

    save_group = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                            run_name, conditions,
                                                                                                            'group', stage)

    save_X1 = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond1', stage)

    save_X2 = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                        run_name, conditions,
                                                                                                        'X-cond2', stage)

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

    arr_scores = classif_single_chan(X_subj, y_subj, group, tmin, tmax)

    return X_subj, y_subj, X_cond1, X_cond2

def prepare_erps(subj_list, tmin, tmax, cond1, cond2, save=True) :
    
    X_cond1 = []
    X_cond2 =[]
    group = []
    y_subj = []
    X_subj = []
    
    for i_subj, subj in enumerate(subj_list) : 
        
        _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
        epochs = mne.read_epochs(path_epochs, verbose=None)
        print(epochs.event_id[cond1])
 
        evokeds_cond1 = epochs[cond1].average()
        evokeds_cond2 = epochs[cond2].average()

        evokeds_cond1_crop = evokeds_cond1.copy().crop(tmin=tmin, tmax=tmax)
        evokeds_cond2_crop = evokeds_cond2.copy().crop(tmin=tmin, tmax=tmax)
        print(evokeds_cond1_crop.data.shape)
        
        # Average time
        print('Average time')
        ave_time_cond1 = np.mean(evokeds_cond1_crop.data, axis=1)
        ave_time_cond2 = np.mean(evokeds_cond2_crop.data, axis=1)

        X_cond1.append(ave_time_cond1)
        X_cond2.append(ave_time_cond2)

        for i_cond in range(2) : 
            group.append(i_subj) # Add subject id to group
        
        y_subj.append(epochs.event_id[cond1]) 
        y_subj.append(epochs.event_id[cond2])


    print('Cond1 shape :', np.array(X_cond1).shape) # Shape should be n_obs, n_chan
    print('Cond2 shape :', np.array(X_cond2).shape) # Shape should be n_obs, n_chan

    print('Concatenation')
    X_subj.append(np.concatenate((np.array(X_cond1), np.array(X_cond2))))

    print(X_subj[-1].shape)
    print(np.array(y_subj).shape)
    print(np.array(group).shape)

    if save == True : 
        save_data(save, tmin, tmax, cond1, cond2, X_subj, y_subj, X_cond1, X_cond2, group)

    arr_scores = classif_single_chan(X_subj, y_subj, group, tmin, tmax) 

    
    return X_subj, y_subj, group, X_cond1, X_cond2

def classif_single_chan(X_subj, y_subj, group,  tmin, tmax) : 
    
    ########## PREPARE DATA ###########
    CHAN = np.arange(0, 270, 1)

    all_scores = []

    # Select the classifier & cross-val method
    X = X_subj[-1] #X_subj[-1]
    print(X.shape)
    X = X*10e26

    y = np.array(y_subj)
    print(y.shape)

    groups = np.array(group)

    ############ CLASSIFICATION #############    
    for chan in CHAN :
        print('-->Process channel :', chan)

        y_pred = []

        clf = RandomForestClassifier()
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        # Number of features to consider at every split
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [2, 5, 10, 15]
        # Minimum number of samples required to split a node
        min_samples_split = [10, 20, 25]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the param grid
        param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        print(param_grid)

        cv = LeaveOneGroupOut() # Cross-validation via LOGO

        # Select channel of interest
        X_chan = X[:, chan]
        X_classif = X_chan.reshape(-1, 1) # Reshape (n_event, 1)
        print(X_classif.shape)

        # Find best params with GridSearch
        # Use RandomSearch
        print('---->Find best params')
        search = RandomizedSearchCV(clf, param_grid, cv=cv, verbose=True, n_jobs=-1, n_iter=50).fit(X=X_classif, 
                                                                                        y=y, 
                                                                                        groups=groups)

        # Select best params
        best_params = search.best_estimator_
        print('Best params : ' + str(best_params))

        # Accuracy score
        scores = search.score(X=X_classif, y=y) 
        print('Scores', scores)

        all_scores.append(scores)

    print(all_scores)

    sub_name = 'all'
    run_name = 'all'
    tmin_name = str(int(tmin*1000))
    tmax_name = str(int(tmax*1000))
    stage = tmin_name + '-' + tmax_name 
    conditions = cond1 + '-' + cond2
    score = 'scores-rdm_forest'

    save_scores = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                                                run_name, conditions,
                                                                                                                                score, stage)
    with open(save_scores, 'wb') as f:
        pickle.dump(all_scores, f)
    
    return all_scores

if __name__ == "__main__" :

    subj_list = SUBJ_CLEAN
    task = args.task

    # Condition 1 and 2
    cond1 = args.condition1
    cond2 = args.condition2
    
    # Time selection
    tmin = args.tmin
    tmax = args.tmax

    # Weither or not to compute hilbert or upload data
    compute_erp = args.compute_erp
    print(compute_erp)

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    # Select what conditions to compute
    condition_list = [cond1, cond2]

    if compute_erp == True : 
        X_subj, y_subj, group = prepare_erps(subj_list, 
                                            tmin, tmax, 
                                            cond1, cond2,
                                            save=True)
    elif compute_erp == False :
        upload_data(tmin, tmax, cond1, cond2)