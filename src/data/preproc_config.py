import matplotlib
from src.params import BIDS_PATH, PREPROC_PATH, ACTIVE_RUN, PASSIVE_RUN
import sys
import os

print(sys.argv)
matplotlib.use('agg')

###############################################################################
# Config parameters
# -----------------

user = os.path.expanduser('~')
interactive = True
on_error = 'debug'

study_name = 'Solaugh'
bids_root = os.path.join(user, 'scratch', 'laughMEG_bids')
deriv_root = os.path.join(user, 'scratch', 'laughter_data', 'preprocessing',
                          'preproc_erp_resample')
random_state = 42

if os.path.exists(deriv_root) == False : 
    os.mkdir(deriv_root)

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
l_freq = 0.1
h_freq = 30. 
notch_freq = [60, 120]
raw_resample_sfreq = 600

# Artifact correction.
spatial_filter = 'ica'
ica_max_iterations = 500
ica_l_freq = 1.
ica_n_components = 0.99
ica_reject_components = 'auto'

# Epochs
epochs_tmin = -0.1
epochs_tmax = 1.5
baseline = (None, 0)

# Conditions / events to consider when epoching
if task == "LaughterActive" : 
    rename_events = {'Bad' : 'Miss'}
    conditions = ['LaughReal', 'LaughPosed', 'Good', 'Miss']
    event_repeated = 'drop'

    # Decoding
    decode = True
    contrasts = [('LaughReal', 'LaughPosed')]

    # Time-frequency analysis 
    # time_frequency_conditions = ['LaughPosed', 'LaughReal', 'Good', 'Miss']
    decoding_csp = False

elif task == "LaughterPassive" :
    
    conditions = ['LaughReal', 'LaughPosed', 'EnvReal', 'ScraReal', 'EnvPosed','ScraPosed'] 
    event_repeated = 'drop'
    
    # Decoding
    decode = True
    contrasts = [('LaughReal', 'LaughPosed')]

    # Time-frequency analysis 
    # time_frequency_conditions = ['EnvReal', 'ScraReal', 'EnvPosed','ScraPosed']
    decoding_csp = False

elif task == 'RS' : 
    task_is_rest = True

# Source reconstruction
run_source_estimation = False
