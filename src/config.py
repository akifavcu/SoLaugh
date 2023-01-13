# TODO : Check if this command works for preprocessing
import matplotlib
matplotlib.use('agg')

###############################################################################
# Config parameters
# -----------------

from src.params import BIDS_PATH, PREPROC_PATH

interactive = True
on_error = 'debug'

study_name = 'Solaugh'
bids_root = BIDS_PATH
deriv_root = PREPROC_PATH

subjects = ['02'] # if 'all' include all subjects
sessions = ['recording']

task = 'LaughterActive'
runs = ['07', '08', '09', '10', '11', '12'] # Active runs = ['07', '08', '09', '10', '11', '12']

find_flat_channels_meg = False
find_noisy_channels_meg = False
use_maxwell_filter = False
ch_types = ['mag']
data_type = 'meg'
eog_channels = ['EEG057', 'EEG058']

# Filtering
mf_cal_fname = None
l_freq = 1.
h_freq = 120. # Need to notch at 60 Hz
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
conditions = ['LaughPosed', 'LaughReal']
event_repeated = 'drop'

# Noise estimation
process_empty_room = False
#noise_cov = 'emptyroom'

# Decoding
decode = True
contrasts = [('LaughReal', 'LaughPosed')]

# Time-frequency analysis 
time_frequency_conditions = ['LaughPosed', 'LaughReal']
decoding_csp = False