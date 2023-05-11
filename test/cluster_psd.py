import mne 
import os
import pickle
import argparse
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from src.utils import get_bids_file, compute_ch_adjacency
from src.params import PREPROC_PATH, FREQS_LIST, FREQS_NAMES, EVENTS_ID, RESULT_PATH, SUBJ_CLEAN, FIG_PATH
from mne.time_frequency import (tfr_morlet, AverageTFR)
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test
from mne.stats import permutation_cluster_1samp_test,  permutation_cluster_test

all_evokeds_list = []
cond1 = 'LaughReal'
cond2 = 'LaughPosed'
task = 'LaughterActive'

contrasts_all_subject = []


print("--> Process task :", task)

path = '/home/claraelk/scratch/laughter_data/results/meg/reports/sub-all/erp/'

print('--> Read epochs condition 1')
epo_name1, _ = get_bids_file(PREPROC_PATH, stage='erp', task=task, condition=cond1)
epochs_cond1 = mne.read_epochs(path + epo_name1)
epochs_cond1.pick_types(meg=True, ref_meg = False,  exclude='bads')

print('--> Read epochs condition 2')
epo_name2, _ = get_bids_file(PREPROC_PATH, stage='erp', task=task, subj=subj, condition=cond2)
epochs_cond2 = mne.read_epochs(path + epo_name2)
epochs_cond2.pick_types(meg=True, ref_meg = False,  exclude='bads')

# Compute freqs from 2 - 60 Hz
print('--> Start TFR morlet')
freqs = np.logspace(*np.log10([2, 60]))
n_cycles = freqs / 2.  # different number of cycle per frequency
power_cond1 = tfr_morlet(epochs_cond1, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                        return_itc=False, decim=3, n_jobs=None, average=False)

power_cond2 = tfr_morlet(epochs_cond2, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                        return_itc=False, decim=3, n_jobs=None, average=False)

print('--> TFR morlet Done')

power_cond1.apply_baseline(mode='ratio', baseline=(None, 0))
power_cond2.apply_baseline(mode='ratio', baseline=(None, 0))

epochs_power_1 = power_cond1.data[:, 0, :, :]  
epochs_power_2 = power_cond2.data[:, 0, :, :]

degrees_of_freedom = len(epochs_power_1) - 1
t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)

print('Computing adjacency.')
adjacency, ch_names = compute_ch_adjacency(all_evokeds.info, ch_type='mag')
print(adjacency.shape)

threshold = 6.0
F_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([epochs_power_1, epochs_power_2], out_type='indices',
                             n_permutations=100, threshold=threshold, tail=0)


T_obs, clusters, cluster_p_values, H0 = cluster_stats

good_cluster_inds = np.where(cluster_p_values < 0.01)[0]
print("Good clusters: %s" % good_cluster_inds)