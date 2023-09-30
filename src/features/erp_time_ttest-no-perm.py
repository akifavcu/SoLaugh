
import os
import mne
import scipy.stats
import matplotlib
import argparse


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from src.params import SUBJ_CLEAN, RESULT_PATH, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, FREQS_LIST, PREPROC_PATH, FREQS_NAMES
from src.utils import get_bids_file
from scipy.stats import ttest_rel
from mne.viz import plot_compare_evokeds
from statsmodels.stats.multitest import fdrcorrection

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
    help="Condition 1 to compute",
)
parser.add_argument(
    "-cond2",
    "--condition2",
    default="LaughPosed",
    type=str,
    help="Condition 2 to compute",
)
parser.add_argument(
    "-cond3",
    "--condition3",
    default="Miss",
    type=str,
    help="Condition 3 to compute",
)
parser.add_argument(
    "-corr",
    "--corr",
    default="FDR",
    type=str,
    help="Correction to apply",
)
args = parser.parse_args()


def prepare_data_ttest(subj_list, task, cond1, cond2, ROI, correction, button_cond) : 

    for roi in ROI : 
        print('--> Compute ROI :', roi)

        evoked_condition1 = []
        evoked_condition2 = []

        evoked_condition1_data = []
        evoked_condition2_data = []

        conditions = cond1 + '-' + cond2

        for subj in subj_list :
            print("processing -->", subj)

            chan_selected = []

            # Load epochs data
            _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj=subj, stage="AR_epo")
            epochs = mne.read_epochs(path_epochs, verbose=None)
            epochs.apply_baseline(baseline=(None, 0))

            CHAN = epochs.info['ch_names']

            # Select channel of interest
            for chan in CHAN  : 
                if roi in chan : 
                    chan_selected.append(chan)

            epochs_copy = epochs.copy() 
            epochs_copy.pick_channels(chan_selected)

            if button_cond == True : # Compute button press : Combine events Good & Bad
                evoked_cond1_roi = mne.combine_evoked([epochs_copy[cond1].average(), epochs_copy[cond3].average()], weights='nave')

            else : # For other conditions
                evoked_cond1_roi = epochs_copy[cond1].average()

            # Create fake evoked with value=0 
            if cond2 == 'BaselineZero' : 
                baseline_data = np.zeros((epochs_copy.get_data().shape[1], epochs_copy.get_data().shape[2]))
                evoked_cond2_roi = mne.EvokedArray(baseline_data, epochs_copy.info, tmin=-0.5, comment='baseline')
            else :
                evoked_cond2_roi = epochs_copy[cond2].average()

            # Prepare data for ttest with roi
            evoked_condition2_data.append(evoked_cond2_roi.get_data()) 
            evoked_condition1_data.append(evoked_cond1_roi.get_data())

            evoked_condition2.append(evoked_cond2_roi) 
            evoked_condition1.append(evoked_cond1_roi)

        # Combine all subject together
        evokeds = {cond1 : evoked_condition1, cond2 : evoked_condition2}

        compute_ttest_times(task, roi, evoked_condition1_data, evoked_condition2_data, 
                            evokeds, evoked_cond1_roi, correction, conditions) 

def compute_ttest_times(task, roi, evoked_condition1_data, evoked_condition2_data, evokeds, 
                        evoked_cond1_roi, correction, conditions) : 
    
    print('--> Compute Ttest')
    # Perform t-test without correction without clusters
    cond1_data = np.array(evoked_condition1_data)
    cond2_data = np.array(evoked_condition2_data)
    print(cond1_data.shape)
    print(cond2_data.shape)

    # Average signal across channel
    cond1_data_ave = np.transpose(np.mean(cond1_data, axis=1), [1, 0])
    cond2_data_ave = np.transpose(np.mean(cond2_data, axis=1), [1, 0])

    print(cond1_data_ave.shape)
    print(cond2_data_ave.shape)

    report = "time={i_ti}, t({df})={t_val:.3f}, p={p:.3f}"
    print("\nTargeted statistical test results:")

    p_vals = []
    x_time = []

    for i_ti, ti in enumerate(cond1_data_ave):

        data1 = cond1_data_ave[i_ti, :]
        data2 = cond2_data_ave[i_ti, :]

        # conduct t test
        t, p = ttest_rel(data1, data2, axis=None) # Paired t-test

        p_vals.append(p)
        x_time.append(i_ti)

        # display results
        format_dict = dict(
            i_ti=evoked_cond1_roi.times[i_ti], df=len(cond1_data_ave) - 1, t_val=t, p=p
        )
        print(report.format(**format_dict))
    
    plot_ttest_evoked(task, roi, p_vals, evokeds, cond1_data_ave, evoked_cond1_roi, correction, conditions)


def plot_ttest_evoked(task, roi, p_vals, evokeds, cond1_data_ave, evoked_cond1_roi, correction, conditions) : 
    
    # Select pvalues sig.
    pval_corrected = 0.05/len(np.transpose(cond1_data_ave, [1, 0])) #Number of time point
    pval_corr = []
    time_corr = []

    su = 'all'
    task = task
    run = 'all'
    cond = conditions
    meas = 'ttest-no-perm-corr' + correction
    stage= roi + '-erp'

    save_fig_path = FIG_PATH + '/erp/ttest_no_cluster/sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}.png'.format(su,
                                                                                                                              task,
                                                                                                                              run,
                                                                                                                              cond,
                                                                                                                              meas,
                                                                                                                              stage)

    if correction == 'Bonferroni' :
        print('Apply Boneferroni correction') # Boneferroni correction
        for i, pval in enumerate(p_vals) : 
            if pval < pval_corrected :
                pval_corr.append(pval)
                time_corr.append(evoked_cond1_roi.times[i])
                
    elif correction == 'FRD' :
        print('Apply FDR correction') # FDR correction
        for i, pval in enumerate(p_vals) :      
            _, pval_corr = fdrcorrection(pval, 
                        alpha=0.05)
            time_corr.append(evoked_cond1_roi.times[i])

    else : # No correction applied
        print("No correction applied")
        for i, pval in enumerate(p_vals) :      
            pval_corr.append(pval)
            time_corr.append(evoked_cond1_roi.times[i])

    colors = "r",'steelblue'

    fig, ax = plt.subplots(1, 1, figsize=(17, 5))

    plot_compare_evokeds(evokeds,
                    colors=colors, show=False, axes=ax,
                    split_legend=True, truncate_yaxis='auto', combine="mean")

    for i, val in enumerate(pval_corr):
        sig_times = []
        
        # plot temporal cluster extent
        sig_times.append(time_corr[i])
        ymin, ymax = ax.get_ylim()
        ax.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                            color='orange', alpha=0.3)
    print('Save figure')
    plt.savefig(save_fig_path)
    plt.show()
    

if __name__ == "__main__" :

    task = args.task
    subj_list = SUBJ_CLEAN

    # Select what conditions to compute (str)
    cond1 = args.condition1
    cond2 = args.condition2
    cond3 = args.condition3
    correction = args.corr

    if cond1 == 'Good' and 'cond3' == 'Miss' :
        button_cond = True
    else :
        button_cond = False
            
    ROI = ['MLT', 'MRT', 'MLP', 'MRP', 'MLC',
            'MRC', 'MLF', 'MRF','MLO','MRO']

    prepare_data_ttest(subj_list, task, cond1, cond2, ROI, correction, button_cond)