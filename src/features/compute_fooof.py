# Import basics
import os
from src.utils import get_bids_file
from src.params import RESULT_PATH, SUBJ_CLEAN 
import pickle
import argparse


# Import MNE, as well as the MNE sample dataset
import mne
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

# FOOOF imports
import fooof
from fooof import FOOOF

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
    "--cond1",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
parser.add_argument(
    "-cond2",
    "--cond2",
    default="LaughPosed",
    type=str,
    help="Condition to compute",
)
args = parser.parse_args()

def compute_foof(task, conditions) : 
    ###### Slope per subjects ######
    cond1_exponent = {}
    cond2_exponent = {}

    for i_cond, cond in enumerate(conditions) :
        print('condition -->', cond)
            
        for subj in SUBJ_CLEAN :
            print('sub-', subj)
            chan_spectrum = []

            all_exponent = []

            # Get path
            _, epo_path = get_bids_file(RESULT_PATH, subj=subj, task=task, stage='AR_epo')

            # Read epochs
            epochs = mne.read_epochs(epo_path)
            epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')

            # Compute PSD
            psd = epochs[cond].compute_psd(fmin=2.0, fmax=200.0)
            print(psd)
            
            # Take freqs
            freqs_subj = psd.freqs
            print(freqs_subj.shape)

            # Take data
            data = psd.get_data()
            data = np.transpose(data, [1, 0, 2]) # Average through epochs 
            print(data.shape)
            
            data_ave = np.mean(data, axis = 1) 
            print(data_ave.shape)
            
            del psd 
            
            for i_chan, chan in enumerate(data_ave) :

                # Organize data 
                chan_spectrum.append(np.array(data_ave[i_chan]))

            print(len(chan_spectrum)) # Length must be n_chan

            for i_spec, spec in enumerate(chan_spectrum) :

                fm = FOOOF()

                # Set the frequency range to fit the model
                freq_range = [2, 200]

                spectrum = data_ave[i_spec]
                print(spectrum.shape) # Size must be n_freqs

                # Report: fit the model, print the resulting parameters, and plot the reconstruction
                fm.fit(freqs=freqs_subj, power_spectrum=spectrum, freq_range=freq_range)
                exponent = fm.get_params('aperiodic_params', 'exponent')

                all_exponent.append(exponent)

            print(len(all_exponent)) # Length should be n_chan
        
            print('Save sub-', subj)
            if i_cond == 0 : 
                cond1_exponent[subj] = all_exponent
            else :
                cond2_exponent[subj] = all_exponent

    # Save files
    filename_cond1 = 'sub-all_task-{}_run-all_cond-{}_meas-fooof-exponent.pkl'.format(task, conditions[0])
    filename_cond2 = 'sub-all_task-{}_run-all_cond-{}_meas-fooof-exponent.pkl'.format(task, conditions[1])

    save_cond1 = os.path.join(RESULT_PATH, 'meg/reports', 'sub-all', 'fooof', filename_cond1) 
    save_cond2 = os.path.join(RESULT_PATH, 'meg/reports', 'sub-all', 'fooof', filename_cond2) 

    with open(save_cond1, 'wb') as f : 
        pickle.dump(cond1_exponent, f)
    
    with open(save_cond2, 'wb') as f : 
        pickle.dump(cond2_exponent, f)

    return cond1_exponent, cond2_exponent, epochs

def ttest_fooof(cond1_exponent, cond2_exponent, epochs, conditions) :
    # Perform t-test without correction without clusters
    cond1_list = []
    cond2_list = []

    for subj in SUBJ_CLEAN: 
        cond1_list.append(cond1_exponent[subj])
        cond2_list.append(cond2_exponent[subj])

        cond1_data = np.array(cond1_list)
        cond2_data = np.array(cond2_list)

        print(cond1_data.shape)
        print(cond2_data.shape)

        cond1_data = np.transpose(cond1_data, [1, 0])
        cond2_data = np.transpose(cond2_data, [1, 0])

        report = "channel={i_ch}, t({df})={t_val:.3f}, p={p:.3f}"
        print("\nTargeted statistical test results:")

        p_vals = []
        x_time = []

        correction = 'FDR'
        ch_corr = []
        pval_corr = []

    for i_ch, ch in enumerate(cond1_data):

        data1 = cond1_data[i_ch, :]
        data2 = cond2_data[i_ch, :]

        # conduct t test
        t, p = ttest_rel(data1, data2, axis=None) # Paired t-test

        p_vals.append(p)
        x_time.append(i_ch)

        # display results
        format_dict = dict(
            i_ch=i_ch, df=len(cond1_data) - 1, t_val=t, p=p
        )
        print(report.format(**format_dict))
        
        if correction == 'FDR' :
            print('Apply FDR correction') # FDR correction

            _, qval = fdrcorrection(p_vals, 
                            alpha=0.05)

            for i_q, q in enumerate(qval) : 
                if q < 0.05 : 
                    ch_corr.append(epochs.info['ch_names'][i_q])
                    pval_corr.append(q) 

    filename_ttest = 'sub-all_task-{}_run-all_cond-{}-{}_meas-fooof-ttest.pkl'.format(task, conditions[0], conditions[1])

    save_ttest = os.path.join(RESULT_PATH, 'meg/reports', 'sub-all', 'fooof', filename_ttest) 

    with open(save_ttest, 'wb') as f : 
        pickle.dump(report, f)

if __name__ == '__main__':
    task = args.task
    cond1 = args.cond1
    cond2 = args.cond2

    conditions = [cond1, cond2]

    cond1_exponent, cond2_exponent, epochs = compute_foof(task, conditions)
    
    ttest_fooof(cond1_exponent, cond2_exponent, epochs, conditions) 

    

