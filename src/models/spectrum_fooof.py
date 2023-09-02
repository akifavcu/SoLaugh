# Import the FOOOF object
import mne
import pickle
import argparse
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
from autoreject import AutoReject
from src.utils import get_bids_file
from src.params import RESULT_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, FIG_PATH, EVENTS_ID

parser = argparse.ArgumentParser()
parser.add_argument(
    "-task",
    "--task",
    default="LaughterActive",
    type=str,
    help="Task to process",
)
parser.add_argument(
    "-cond",
    "--condition",
    default="LaughReal",
    type=str,
    help="Condition to compute",
)
args = parser.parse_args()

def prepare_psd_data(SUBJ_CLEAN, task, conditions, event_id, run_list):
    all_subj_psd = []

    for i, cond in enumerate(conditions) :
        print('condition -->', cond)
        all_psd = []
        freqs_list = []

        for subj in SUBJ_CLEAN :
            
            # Take ica data
            _, ica_path = get_bids_file(PREPROC_PATH, stage='ica', subj=subj)
            ica = mne.preprocessing.read_ica(ica_path)

            for run in run_list : 

                # Get path
                _, epo_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, run=run, measure='clean-', stage='epo')

                epochs_clean = mne.read_epochs(epo_path, preload=True)

                # Compute PSD
                psd = epochs_clean[cond].compute_psd(fmin=1.0, fmax=120.0)

                # Take freqs
                freqs_subj = psd.freqs
                freqs_list.append(freqs_subj)
                
                # Take data
                data = psd.get_data()
                all_psd.append(data)

    # Average frequency across subjects
    freqs = np.mean(np.array(freqs_list), axis=0)
    print(freqs.shape)

    # Average power across :
    print(all_psd.shape())
    # runs
    psd_ave_run = np.mean(np.array(all_psd), axis=0)
    # epochs
    psd_ave_epo = np.mean(np.array(psd_ave_run), axis=0)
    # channels
    psd = np.mean(np.array(psd_ave_epo), axis=0)

    print(psd.shape)

    #_, save_path = get_bids_file(RESULT_PATH, stage='psd', measure='log_fooof', task=task, condition=cond)

    #with open(save_path, 'wb') as f:
    #    pickle.dump(psd, f)  
    
    apply_fooof(psd, task=task, condition=cond, freqs=freqs)

    return psd

def apply_fooof(psd, task, condition, freqs) :

    fooof_path = os.path.join(RESULT_PATH, 'meg', 'reports', 'figures', 'fooof')
    freq_range = [2, 120]
    
    # Fit model to data
    fm = FOOOF()
    fm.fit(freqs, psd, freq_range)  

    # Save data
    fm.save('subj-all_task-{}_cond-{}_fooof_data_epo3sec'.format(task, condition), 
            fooof_path, save_results=True, save_settings=True, save_data=True)

    # Save fooof report
    fm.save_report('subj-all_task-{}_cond-{}_fooof_report_epo3sec'.format(task, condition), fooof_path)

def exponent_per_channel(task, conditions) :
    # Test FOOOF per channel

    for i, cond in enumerate(conditions) :
        
        print('condition -->', cond)
        all_exponent = []
        chan_spectrum = []
        
        all_psd = []
        freqs_list = []

        for subj in SUBJ_CLEAN :
            list_epochs = []

            # Get path
            _, epo_path = get_bids_file(PREPROC_PATH, subj=subj, task=task, run=run, measure='clean-', stage='epo')

            epochs = mne.read_epochs(epo_path, preload=True)

            epochs.apply_baseline((None, 0))
            epochs.pick_types(meg=True, ref_meg=False,  exclude='bads')

            # Compute PSD
            psd = epochs[cond].compute_psd(fmin=1.0, fmax=120.0)

            # Take freqs
            freqs_subj = psd.freqs
            freqs_list.append(freqs_subj)

            # Take data
            data = psd.get_data()
            data = np.transpose(data, [1, 0, 2])
            data_ave = np.mean(data, axis = 1)
            # data_ave = np.mean(data_ave, axis = 1)
            all_psd.append(data_ave)

        # Average frequency across subjects
        freqs = np.mean(np.array(freqs_list), axis=0)
        print(freqs.shape)
        
        # Average spectrum accros usbject
        ave_spectra_cond1 = np.mean(np.array(all_psd), axis=0)
        print(ave_spectra_cond1.shape)
        
        for i, chan in enumerate(ave_spectra_cond1) :
            
            # Organize data 
            chan_spectrum.append(np.array(ave_spectra_cond1[i]))
            
        print(len(chan_spectrum)) # Length must be n_chan
        
        for i, spec in enumerate(chan_spectrum) :

            fm = FOOOF()

            # Set the frequency range to fit the model
            freq_range = [2, 120]

            spectrum = chan_spectrum[i]
            print(spectrum.shape) # Size must be 76

            # Report: fit the model, print the resulting parameters, and plot the reconstruction
            fm.fit(freqs=freqs, power_spectrum=spectrum, freq_range=freq_range)
            exponent = fm.get_params('aperiodic_params', 'exponent')

            all_exponent.append(exponent)
        
        print(len(all_exponent)) # Length should be n_chan

        ######## PLOT TOPOMAPS ########

        # initialize figure
        fig, ax_topo = plt.subplots(1)

        image, _ = mne.viz.plot_topomap(np.array(all_exponent), epochs.info, axes=ax_topo, show=True, 
                                cmap='bwr', extrapolate='head',
                                sphere=(0, 0.0, 0, 0.19), cnorm = matplotlib.colors.CenteredNorm(vcenter=0))

        plt.colorbar(image)
        plt.savefig(FIG_PATH + 'fooof/sub-all_task-{}_run-all_cond-{}_meas-fooof_topomap_epo3sec.png'.format(task, cond))

if __name__ == "__main__" :
    task = args.task
    # cond = args.condition

    # conditions = [cond]

    if task == 'LaughterActive':
        run_list = ACTIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']
        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66, }

    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN
        conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'EnvPosed',
                      'ScraReal', 'ScraPosed']
        event_id = {'LaughReal' : 11, 'LaughPosed' : 12, 'EnvReal' : 21, 'ScraReal' : 31, 'EnvPosed' : 22,
            'ScraPosed' : 32,}


    psd = prepare_psd_data(SUBJ_CLEAN, task, conditions, event_id, run_list)
    exponent_per_channel(task, conditions)

