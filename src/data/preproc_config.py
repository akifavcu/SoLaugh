import matplotlib
from src.params import BIDS_PATH, PREPROC_PATH, ACTIVE_RUN, PASSIVE_RUN
import sys

print(sys.argv)
matplotlib.use('agg')

###############################################################################
# Config parameters
# -----------------

interactive = True
on_error = 'debug'

study_name = 'Solaugh'
bids_root = BIDS_PATH
deriv_root = PREPROC_PATH

# Take arguments if exist
list_argv = sys.argv

arg_task = "LaughterActive" #task by default
subj = ['03']  # subject by default

for argument in list_argv :
    if "--subject=" in argument :
        subj = argument[-2:]

    if "--task=" in argument :
        arg_task = argument[7:]

print("Process subject :", subj, "& compute task :", arg_task)

subjects = subj 
sessions = ['recording']

task = arg_task # task by default

if arg_task == "LaughterActive" :
    runs = ACTIVE_RUN
elif arg_task == "LaughterPassive" :
    runs = PASSIVE_RUN

find_flat_channels_meg = False
find_noisy_channels_meg = False
use_maxwell_filter = False
ch_types = ['mag']
data_type = 'meg'
eog_channels = ['EEG057', 'EEG058']

# Noise estimation
process_empty_room = False
#noise_cov = 'emptyroom'

# Filtering
mf_cal_fname = None
l_freq = 1.
h_freq = 120. # Need to notch at 60 Hz
notch_freq = [60, 120]
#raw_resample_sfreq = 250

# Artifact correction.
spatial_filter = 'ica'
ica_max_iterations = 500
ica_l_freq = 1.
ica_n_components = 0.99
ica_reject_components = 'auto'


# Epochs
epochs_tmin = -0.5
epochs_tmax = 1.5
baseline = (None, 0)

# Conditions / events to consider when epoching
if task == "LaughterActive" : 
    rename_events = {'LaughReal' : 'Laugh/Real', 'LaughPosed' : 'Laugh/Posed', 
                    'Bad' : 'Miss'}
    conditions = ['Laugh/Real', 'Laugh/Posed', 'Good', 'Miss']
    event_repeated = 'drop'

    # Decoding
    decode = False
    # contrasts = [('LaughReal', 'LaughPosed')]

    # Time-frequency analysis 
    # time_frequency_conditions = ['LaughPosed', 'LaughReal']
    decoding_csp = False

elif task == "LaughterPassive" :
    if runs == '02' or runs == '03' :
        rename_events = {'LaughReal' : 'Laugh/Real', 'LaughPosed' : 'Laugh/Posed', 
                        'EnvReal' : 'Env/Real', 'EnvPosed' : 'Env/Posed'}
        conditions = ['Laugh/Real', 'Laugh/Posed']
        event_repeated = 'drop'
    elif runs == '04' or runs == '05':
        rename_events = {'LaughReal' : 'Laugh/Real', 'LaughPosed' : 'Laugh/Posed', 
                        'ScraReal' : 'Scra/Real', 'ScraPosed' : 'Scra/Posed'}
        conditions = ['Laugh/Real', 'Laugh/Posed']
        event_repeated = 'drop'
    
    # Decoding
    decode = False
    # contrasts = [('EnvReal', 'EnvPosed')]

    # Time-frequency analysis 
    # time_frequency_conditions = ['EnvReal', 'ScraReal', 'EnvPosed','ScraPosed']
    decoding_csp = False

# Source reconstruction
run_source_estimation = False
