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
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut, cross_val_score, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression, LinearRegression



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
    default='True',
    type=str,
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

    return X_subj, y_subj, group, X_cond1, X_cond2

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
    
    CHAN = np.arange(0, 270, 1)
    norm = 1
    randomSearch=False

    all_scores = {}
    score_list = []
    perm_list = []
    pval_list = []

    # Select the classifier & cross-val method
    X = X_subj[-1]
    print(X.shape)

    y = np.array(y_subj)
    print(y.shape)

    groups = np.array(group)

    if randomSearch == False : 
        ############ CLASSIFICATION #############    
        for chan in CHAN :
            print('-->Process channel :', chan)

            clf = GradientBoostingClassifier()
            cv = LeaveOneGroupOut() # Cross-validation via LOGO

            if norm == 1:
                scaler = StandardScaler()
                pipeline = Pipeline([("scaler", scaler), ("classifier", clf)])
            else:
                pipeline = clf

            # Select channel of interest
            X_chan = X[:, chan]
            X_classif = X_chan.reshape(-1, 1) # Reshape (n_event, 1)
            print(X_classif.shape)

            # Find best params with GridSearch
            # Use RandomSearch
            print('---->Find best params')
            scores, permutation_scores, p_value = permutation_test_score(pipeline, X_classif, y, cv=cv, 
                                                                         groups=groups, n_jobs=-1)
            print(scores)
            print(permutation_scores)
            print(p_value)

            score_list.append(scores)
            perm_list.append(permutation_scores)
            pval_list.append(p_value)

        all_scores['score'] = score_list
        all_scores['permutation'] = perm_list
        all_scores['pval'] = pval_list

        print('Save file')
        sub_name = 'all'
        run_name = 'all'
        tmin_name = str(int(tmin*1000))
        tmax_name = str(int(tmax*1000))
        stage = tmin_name + '-' + tmax_name 
        conditions = cond1 + '-' + cond2
        score = 'scores-xgboost-perm'

        save_scores = RESULT_PATH + 'meg/reports/sub-all/ml/erp/sub-{}_task-{}_run-{}_cond-{}_meas-sf-{}_{}.pkl'.format(sub_name, task, 
                                                                                                                        run_name, conditions,
                                                                                                                        score, stage)
        with open(save_scores, 'wb') as f:
            pickle.dump(all_scores, f)
            
    else : 

        ############ CLASSIFICATION #############    
        for chan in CHAN :

            print('-->Process channel :', chan)

            clf =  RandomForestClassifier()
            cv = LeaveOneGroupOut() # Cross-validation via LOGO

            # Select channel of interest
            X_chan = X[:, chan]
            X_classif = X_chan.reshape(-1, 1) # Reshape (n_event, 1)
            print(X_classif.shape)

            # Define the parameter grid for RandomizedSearchCV
            '''param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear']
            }

            param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],}'''
            
            param_grid = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__bootstrap': [True, False]
        }


            # Create the pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Standardization
                ('classifier', clf),  #  Classifier
            ])
            # Create the RandomizedSearchCV
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                scoring='accuracy',  # Change this to the appropriate metric
                cv=cv.split(X_classif, groups=groups),  
                n_iter=10,  
                n_jobs=-1  
            )

            # Fit the pipeline with RandomizedSearchCV
            random_search.fit(X_classif, y)

            # Access the best estimator and its parameters
            best_pipeline = random_search.best_estimator_
            best_params = random_search.best_params_

            # Now, use permutation_test_score
            scores, permutation_scores, p_value = permutation_test_score(
                best_pipeline, X_classif, y, groups=groups, cv=cv.split(X_classif, groups=groups), n_permutations=100, n_jobs=-1)

            # Display the results
            print("Best Parameters: ", best_params)
            print("Permutation Test Score: ", scores)
            score_list.append(scores)
            perm_list.append(permutation_scores)
            pval_list.append(p_value)
        
        all_scores['score'] = score_list
        all_scores['permutation'] = perm_list
        all_scores['pval'] = pval_list

        print('Save file')
        sub_name = 'all'
        run_name = 'all'
        tmin_name = str(int(tmin*1000))
        tmax_name = str(int(tmax*1000))
        stage = tmin_name + '-' + tmax_name 
        conditions = cond1 + '-' + cond2
        score = 'scores-rdm_forest-perm-rdmSearch'

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

    if compute_erp == 'True' : 
        X_subj, y_subj, group, X_cond1, X_cond2 = prepare_erps(subj_list, 
                                            tmin, tmax, 
                                            cond1, cond2,
                                            save=True)
    elif compute_erp == 'False' :
        upload_data(tmin, tmax, cond1, cond2)