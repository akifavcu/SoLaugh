import mne
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_bids_file
from src.params import BIDS_PATH, PREPROC_PATH, SUBJ_CLEAN, ACTIVE_RUN, PASSIVE_RUN, RESULT_PATH, EVENTS_ID, FIG_PATH, FREQS_LIST, FREQS_NAMES
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.backends.backend_pdf import PdfPages 


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
parser.add_argument(
    "-freq",
    "--freq",
    default="alpha",
    type=str,
    help="Frequency to compute",
)
args = parser.parse_args()

def main(task, run_list, cond1, cond2, freq) : 

    ROI = ['MRT44', 'MLT44',
        'MRT53', 'MLT53',
        'MRT54', 'MLT54']
    #['MLO', 'MRO',
    #'MLP', 'MRP',
    #'MLT', 'MRT',
    #'MLC', 'MRC',
    #'MLF', 'MRF',
    #'MZ', 'MZC']

    _, path_epochs = get_bids_file(RESULT_PATH, task=task, subj='01', stage="AR_epo")
    epochs = mne.read_epochs(path_epochs, verbose=None)

    for freq_name in [freq]:
        print('Processing freqs -->', freq_name)

        contrasts_all_subject = []
        
        filename = 'envelop_spectral/sub-all_task-{}_run-all_cond-{}-{}_meas-envelop-one-chan_freq-{}.pdf'.format(task, cond1, 
                                                                                                        cond2, freq_name)
        pdf = PdfPages(FIG_PATH + filename)
        print(filename)

        for roi in ROI :
            
            chan_selected = []
            evokeds_cond1 = []
            evokeds_cond2 = []
            
            for chan in epochs.info['ch_names'] : 
                if roi in chan : 
                    chan_selected.append(chan)
                    
            print(chan_selected)

            for i_subj, subj in enumerate(subj_list):
                print("=> Process task :", task, 'subject', subj)

                epochs_list = []
                evoked_condition1_data = []
                evoked_condition2_data = []

                sub_id = 'sub-' + subj
                subj_path = os.path.join(RESULT_PATH, 'meg', 'reports', sub_id, 'hilbert/')

                if not os.path.isdir(subj_path):
                    os.mkdir(subj_path)
                    print('BEHAV folder created at : {}'.format(subj_path))
                else:
                    print('{} already exists.'.format(subj_path))

                # Select event_id appropriate
                for run in run_list :
                    print("=> Process run :", run)

                    psd_filename, psd_path = get_bids_file(subj_path, 
                                            stage='hilbert_env-psd_epo', 
                                            subj=subj, 
                                            task=task, 
                                            run=run, 
                                            measure=freq_name)

                    epochs = mne.read_epochs(subj_path + psd_filename)
                    evokeds_cond1.append(epochs[cond1].average(picks=chan_selected))
                    evokeds_cond2.append(epochs[cond2].average(picks=chan_selected))

                    if subj == subj_list[0] and run == run_list[0]:
                        head_info = epochs.info['dev_head_t']
                    else:
                        epochs.info['dev_head_t'] = head_info

                    #epochs.equalize_event_counts([cond1, cond2])

                    epochs_list.append(epochs)

                epochs_all_run = mne.concatenate_epochs(epochs_list)  
            
            ave_channels = epochs_all_run[cond1].average(picks=chan_selected)
            
            fig, ax_topo = plt.subplots(1, 1, figsize=(20, 10))
            ave_channels.plot_sensors(show_names=False, axes=ax_topo, title=roi,
                                    sphere=(0, 0.01, 0, 0.184), show=False)
            
            divider = make_axes_locatable(ax_topo)

            evokeds = {cond1 : evokeds_cond1, 
                    cond2 : evokeds_cond2}
            
            ax_signals = divider.append_axes('right', size='300%', pad=1.2)

            title = roi + ' Conditions ' + cond1 + '-' + cond2 + 'Freq ' + freq_name
            mne.viz.plot_compare_evokeds(evokeds, title=title, axes=ax_signals,
                                        show=False, split_legend=True, 
                                        truncate_yaxis='auto', combine="mean")
            mne.viz.tight_layout(fig=fig)
            fig.subplots_adjust(bottom=.05)
        
            plt.savefig(pdf, format='pdf') 
            plt.show()
        
        pdf.close()
        
if __name__ == "__main__" :

    task = args.task
    subj_list = SUBJ_CLEAN

    # Select what conditions to compute (str)
    cond1 = args.cond1
    cond2 = args.cond2
    freq = args.freq

    if task == 'LaughterActive' :
        run_list = ACTIVE_RUN
    elif task == 'LaughterPassive':
        run_list = PASSIVE_RUN

    main(task, run_list, cond1, cond2, freq)

