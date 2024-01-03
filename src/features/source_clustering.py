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
parser.add_argument(
    "-compute_cluster",
    "--compute_cluster",
    default='True',
    type=str,
    help="Compute cluster",
)

args = parser.parse_args()

def prepare_source_erp(subj_list, task, cond1, cond2, save=True) :

    ####### PREPARE DATA #######

    # Loop through subjects
    all_stc_cond1 = []
    all_stc_cond2 = []

    task = task
    cond1_name = cond1
    cond2_name = cond2

    for subj in subj_list:
        _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
        epochs = mne.read_epochs(path_epochs, verbose=None)

        # Find condition related to epoch
        cond1_epo = []
        cond2_epo = []
        
        all_stc_cond1 = []

        all_stc_cond2 = []
        for i, epo_arr in enumerate(epochs.events): 
            if epo_arr[2] == 3 :                
                cond1_epo.append(i)
            elif epo_arr[2] == 2 : 
                cond2_epo.append(i)

        cond1 = [str(nb).zfill(3) for nb in cond1_epo]
        cond2 = [str(nb).zfill(3) for nb in cond2_epo]
        print('Cond1', len(cond1))
        print('Cond2', len(cond2))
        
        # Find morph data of the same condition
        for epo1 in cond1 : # Cond1
            stc_path_cond1 = os.path.join(MRI_PATH, f'sub-{subj}', 'morph',
                                        f'{epo1}_MNE_morph-stc.h5')

            stc_cond1 = mne.read_source_estimate(stc_path_cond1)
            all_stc_cond1.append(stc_cond1)

        for epo2 in cond2 : # Cond2
            stc_path_cond2 = os.path.join(MRI_PATH, f'sub-{subj}', 'morph',
                                        f'{epo2}_MNE_morph-stc.h5')

            stc_cond2 = mne.read_source_estimate(stc_path_cond2)
            all_stc_cond2.append(stc_cond2)

        # Average 1 suibject for each condition
        grand_ave_cond1 = np.mean(all_stc_cond1) 
        grand_ave_cond2 = np.mean(all_stc_cond2) 
        
        print(grand_ave_cond1.data.shape)
        print(grand_ave_cond2.data.shape)

    # Save file
    if save == True :
        print('Save contrast, cond1 & cond2')
        conditions = cond1_name + "-" + cond2_name

        filename_cond1, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-src', condition=cond1_name)
        filename_cond2, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-src', condition=cond2_name)

        save_cond1 = os.path.join(MRI_PATH, f'sub-{subj}', filename_cond1[:-4])
        save_cond2 = os.path.join(MRI_PATH, f'sub-{subj}', filename_cond2[:-4])

        grand_ave_cond1.save(save_cond1, ftype='h5', overwrite=True)
        grand_ave_cond2.save(save_cond2, ftype='h5',  overwrite=True)


    return grand_ave_cond1, grand_ave_cond2, conditions

def cluster_source(task, cond1_name, cond2_name) :

    contrast_data = []
    group_all_stc_cond1 = []
    group_all_stc_cond2 =[]
    conditions = cond1_name + "-" + cond2_name

    crop = True
    tmin = 0.0
    tmax = 1.5

    ##### Import data
    for subj in SUBJ_CLEAN :
        filename_cond1, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-src', condition=cond1_name)
        filename_cond2, _ = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='erp-morph-src', condition=cond2_name)

        save_cond1 = os.path.join(MRI_PATH, f'sub-{subj}', filename_cond1[:-4])
        save_cond2 = os.path.join(MRI_PATH, f'sub-{subj}', filename_cond2[:-4])

        stc_cond1 = mne.read_source_estimate(save_cond1)
        stc_cond2 = mne.read_source_estimate(save_cond2)

        if crop == True : 
            stc_cond1.crop(tmin, tmax)
            stc_cond2.crop(tmin, tmax)

        group_all_stc_cond1.append(stc_cond1.data)
        group_all_stc_cond2.append(stc_cond2.data)

    ###### Compute contrast
    contrast = np.array(group_all_stc_cond1)-np.array(group_all_stc_cond2) 
    print('Shape contrast :', contrast.shape)

    ###### Compute adjacency #######
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


    ########### STATISTICAL ANALYSIS ##########

    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.
    p_threshold = 0.001
    df = n_subjects - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    n_perm = 1024

    # Now let's actually do the clustering. This can take a long time...
    print("Clustering.")
    T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_jobs=-1,
        threshold=None,
        buffer_size=None,
        verbose=True,
        n_permutations= n_perm,
        tail=0,
        check_disjoint=False,
        out_type='indices'
    )
    
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    print(good_clusters)

    print('Saving cluster')
    if crop == False : 
        save_cluster_stats, _ = get_bids_file(RESULT_PATH, stage = "erp-src", task=task, 
                                              measure=f"Ttest-clusters-morph-perm-{n_perm}", 
                                              condition = conditions)

    else :
        save_cluster_stats, _ = get_bids_file(RESULT_PATH, stage = "erp-src", task=task, 
                                              measure=f"Ttest-clusters-morph-perm-{n_perm}_time{tmin}-{tmax}", 
                                              condition = conditions)
        
    path_save_cluster = os.path.join(MRI_PATH, 'sub-all', save_cluster_stats)

    with open(path_save_cluster, 'wb') as f:
        pickle.dump(clu, f)  

    print('Save done - no disjoint')

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

    compute_cluster = args.compute_cluster
    print(compute_cluster)
    
    print("=> Process task :", task, "for conditions :", cond1, "&", cond2)

    # Compute ERP clusters
    if compute_cluster == 'False' : 
        grand_ave_cond1, grand_ave_cond2, conditions = prepare_source_erp(subj_list, task, cond1, cond2, save)

    elif compute_cluster == 'True': 
        clu = cluster_source(task, cond1, cond2)