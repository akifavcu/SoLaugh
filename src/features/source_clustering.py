import os
import mne
import pickle
import argparse

from src.params import MRI_PATH, SUBJ_CLEAN, RESULT_PATH, SUBJ_CLEAN
from src.utils import get_bids_file

import numpy as np
from scipy import stats as stats
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-subj",
    "--subj",
    default="01",
    type=str,
    help="Subject to process",
)

parser.add_argument(
    "-cond1",
    "--condition1",
    default="LaughReal",
    type=str,
    help="First condition",
)

parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Second condition",
)

parser.add_argument(
    "-save",
    "--save",
    default=True,
    type=bool,
    help="Save file",
)
args = parser.parse_args()

def prepare_source_erp(subj_list, task, cond1, cond2, save=True) :

    ####### PREPARE DATA #######

    # Loop through subjects
    all_subject_stcs_cond1 = []
    all_subject_stcs_cond2 = []

    all_stc_cond1 = []
    all_stc_cond2 = []

    group_all_stc_cond1 = []
    group_all_stc_cond2 = []

    task = task
    cond1_name = cond1
    cond2_name = cond2

    for subj in subj_list:
        _, epo_path = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
        epochs = mne.read_epochs(epo_path)

        # Find condition related to epoch
        cond1_epo = []
        cond2_epo = []
        for i, epo_arr in enumerate(epochs.events): 
            if epo_arr[2] == 3 :                
                cond1_epo.append(i)
            elif epo_arr[2] == 2 : 
                cond2_epo.append(i)

        cond1 = [str(nb).zfill(3) for nb in cond1_epo]
        cond2 = [str(nb).zfill(3) for nb in cond2_epo]
        print('Cond1', len(cond1))
        print('Cond2', len(cond2))

        for epo1 in cond1 : 
            stc_path_cond1 = os.path.join(MRI_PATH, f'sub-{subj}', 'src',
                                        f'{epo1}_MNE_source-stc.h5')
            
            stc_cond1 = mne.read_source_estimate(stc_path_cond1)
            all_stc_cond1.append(stc_cond1)

        for epo2 in cond2 : 
            stc_path_cond2 = os.path.join(MRI_PATH, f'sub-{subj}', 'src',
                                        f'{epo2}_MNE_source-stc.h5')
            
            stc_cond2 = mne.read_source_estimate(stc_path_cond2)
            all_stc_cond2.append(stc_cond2)

        print(len(all_stc_cond1)) 
        print(len(all_stc_cond2))  

        ######## AVERAGE SIGNAL PER CONDITION FOR 1 SUBJ #########
        for subject_stcs1 in all_stc_cond1 : 

            # Average the SourceEstimates for the current subject
            avg_data_cond1 = np.mean([s.data for s in all_stc_cond1], axis=0)
            avg_stc_cond1 = mne.SourceEstimate(avg_data_cond1, 
                                            vertices=stc_cond1.vertices, 
                                            tmin=stc_cond1.tmin, 
                                            tstep=stc_cond1.tstep, 
                                            subject='sub-all')

            # Append to the list for all subjects
            all_subject_stcs_cond1.append(avg_stc_cond1)

        print('All subj cond1 :', len(all_subject_stcs_cond1))


        for subject_stcs2 in all_stc_cond2 : 

            # Average the SourceEstimates for the current subject
            avg_data_cond2 = np.mean([s.data for s in all_stc_cond2], axis=0)
            avg_stc_cond2 = mne.SourceEstimate(avg_data_cond2, 
                                            vertices=stc_cond2.vertices, 
                                            tmin=stc_cond2.tmin, 
                                            tstep=stc_cond2.tstep, 
                                            subject='sub-all')

            # Append to the list for all subjects
            all_subject_stcs_cond2.append(avg_stc_cond2)

        print('All subj cond2 :', len(all_subject_stcs_cond2))

        # Average 1 subject CONDITION 1
        group_avg_data_cond1 = np.mean([s.data for s in all_subject_stcs_cond1], axis=0)
        group_avg_stc_cond1 = mne.SourceEstimate(group_avg_data_cond1, 
                                                vertices=avg_stc_cond1.vertices, 
                                                tmin=avg_stc_cond1.tmin, 
                                                tstep=avg_stc_cond1.tstep, 
                                                subject=avg_stc_cond1.subject)

        # Average 1 subject CONDITION 2
        group_avg_data_cond2 = np.mean([s.data for s in all_subject_stcs_cond2], axis=0)
        group_avg_stc_cond2 = mne.SourceEstimate(group_avg_data_cond2, 
                                                vertices=avg_stc_cond2.vertices, 
                                                tmin=avg_stc_cond2.tmin, 
                                                tstep=avg_stc_cond2.tstep, 
                                                subject=avg_stc_cond2.subject)
        
        group_all_stc_cond1.append(group_avg_stc_cond1.data)
        group_all_stc_cond2.append(group_avg_stc_cond2.data)
        
    print(np.array(group_all_stc_cond1).shape)
    print(np.array(group_all_stc_cond2).shape)
    
    contrast = np.array(group_all_stc_cond1)-np.array(group_all_stc_cond2) 
    
    # Save file
    if save == True :
        print('Save contrast, cond1 & cond2')
        conditions = cond1_name + "-" + cond2_name

        filename_contrast, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-src-contrast', condition=conditions)
        filename_cond1, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-src-contrast', condition=cond1_name)
        filename_cond2, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-src-contrast', condition=cond2_name)

        save_contrasts = os.path.join(MRI_PATH, 'sub-all', filename_contrast)

        save_cond1 = os.path.join(MRI_PATH, 'sub-all', filename_cond1)
        save_cond2 = os.path.join(MRI_PATH, 'sub-all', filename_cond2)

        with open(save_contrasts, 'wb') as f:
            pickle.dump(contrast, f)  

        with open(save_cond1, 'wb') as f:
            pickle.dump(group_all_stc_cond1, f)  

        with open(save_cond2, 'wb') as f:
            pickle.dump(group_all_stc_cond2, f) 

    return contrast, conditions

def cluster_source(task, cond1_name, cond2_name) :

    contrast_data = []
    group_all_stc_cond1 = []
    group_all_stc_cond2 =[]
    conditions = cond1_name + "-" + cond2_name

    ##### Import data
    for subj in SUBJ_CLEAN :
        filename_contrast, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-contrast', condition=conditions)
        filename_cond1, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-contrast', condition=cond1_name)
        filename_cond2, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-contrast', condition=cond2_name)

        save_contrasts = os.path.join(MRI_PATH, 'sub-all', filename_contrast)

        save_cond1 = os.path.join(MRI_PATH, 'sub-all', filename_cond1)
        save_cond2 = os.path.join(MRI_PATH, 'sub-all', filename_cond2)

        with open(save_contrasts, 'rb') as f:
            contrast_subj = pickle.load(f)  

        with open(save_cond1, 'rb') as f:
            group_all_stc_cond1_subj = pickle.load(f)  

        with open(save_cond2, 'rb') as f:
            group_all_stc_cond2_subj = pickle.load(f) 
    
        contrast_data.append(contrast_subj)
        group_all_stc_cond1.append(group_all_stc_cond1_subj[0])
        group_all_stc_cond2.append(group_all_stc_cond2_subj[0])

    ###### Compute contrast
    contrast = np.array(group_all_stc_cond1)-np.array(group_all_stc_cond2) 
    print('Shape contrast :', contrast.shape)

    ###### Compute adjacency
    src_fname = os.path.join(MRI_PATH, 'anat/subjects', 'fsaverage/bem', 'fsaverage-5-src.fif')

    src = mne.read_source_spaces(src_fname)
    print(type(src))
    
    # Find adjacency
    print("Computing adjacency.")
    adjacency = mne.spatial_src_adjacency(src, dist=None)
    print(adjacency.shape)

    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    X = np.transpose(contrast, [0, 2, 1])
    print(X.shape)
    n_subjects = len(contrast)
    print(n_subjects)

    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.
    p_threshold = 0.001
    df = n_subjects - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    # Now let's actually do the clustering. This can take a long time...
    print("Clustering.")
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_jobs=-1,
        threshold=t_threshold,
        buffer_size=None,
        verbose=True,
        n_permutations=1024,
        tail=0,
        step_down_p = 0.05,
        check_disjoint=True,
        out_type='indices'
    )
    
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print(good_clusters)

    print('Saving cluster')
    save_cluster_stats, _ = get_bids_file(RESULT_PATH, stage = "erp-source", task=task, measure="Ttest-clusters-threshold", condition = conditions)

    path_save_cluster = os.path.join(MRI_PATH, 'sub-all', save_cluster_stats)

    with open(path_save_cluster, 'wb') as f:
        pickle.dump(clu, f)  

    print('Save done')

    return clu

if __name__ == "__main__" :
    
    # Select subjects and runs and stage
    task = args.task
    subj_list = [args.subj]

    # Select what conditions to compute (str)
    save = args.save
    cond1 = args.condition1
    cond2 = args.condition2
    condition_list = [cond1, cond2]

    compute_cluster = False
    
    print("=> Process task :", task, "for conditions :", cond1, "&", cond2)

    # Compute ERP clusters
    if compute_cluster == False : 
        contrast, conditions = prepare_source_erp(subj_list, task, cond1, cond2, save)

    elif compute_cluster == True: 
        clu = cluster_source(task, cond1, cond2)