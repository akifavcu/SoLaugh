import os
import sys
import numpy as np
import mne

# Set paths
user = os.path.expanduser('~')
scratch_folder = os.path.join(user, "scratch")
proj_path = os.path.join(user, "SoLaugh/")
study_path = os.path.join(user, "scratch", 'laughter_data') # /home/clara_elk/Documents/
weights_path = os.path.join(scratch_folder, "net_weights")
results_path = os.path.join(study_path, "results")
similarity_folder = os.path.join(study_path, 'similarity_scores')
activations_folder = os.path.join(study_path, 'activations')
rdms_folder = os.path.join(study_path, 'networks_rdms')
plots_path = os.path.join(scratch_folder, "results", "meg", "reports", "figures")
subjects_dir = os.path.join(study_path, 'results', 'mri/anat/', 'subjects')
meg_dir = os.path.join(study_path, 'preproc_bids_data')

# Check and create necessary directories
for folder in [scratch_folder, study_path, results_path,
               subjects_dir, meg_dir]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# Set environment variable
os.environ["SUBJECTS_DIR"] = subjects_dir

# MNE related parameters
spacing = 'oct6'
mindist = 5
N_JOBS = 1
ylim = {'eeg': [-10, 10], 'mag': [-300, 300], 'grad': [-80, 80]}
l_freq = None
tmin = -0.2
tmax = 2.9
reject_tmax = 0.8
random_state = 42
smooth = 10
fsaverage_vertices = [np.arange(10242), np.arange(10242)]
ctc = os.path.join(os.path.dirname(__file__), 'ct_sparse.fif')
cal = os.path.join(os.path.dirname(__file__), 'sss_cal.dat')

# Mapping between filenames and subjects/conditions
map_subjects = {
    1: 'subject_02', 2: 'subject_03', 3: 'subject_06', 4: 'subject_08',
    5: 'subject_09', 6: 'subject_10', 7: 'subject_11', 8: 'subject_12',
    9: 'subject_14', 10: 'subject_15', 11: 'subject_17', 12: 'subject_18',
    13: 'subject_19', 14: 'subject_23', 15: 'subject_24', 16: 'subject_25'
}

conditions_mapping = {}
for i in range(1, 151):
    conditions_mapping[f"meg/f{i:03d}.bmp"] = str(i)
    conditions_mapping[f"meg/u{i:03d}.bmp"] = str(i + 150)
    conditions_mapping[f"meg/s{i:03d}.bmp"] = str(i + 300)

# Plotting parameters and functions
def set_matplotlib_defaults():
    import matplotlib.pyplot as plt
    fontsize = 8
    params = {'axes.labelsize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2,
              'figure.max_open_warning': 200,
              'axes.spines.top': False,
              'axes.spines.right': False,
              'axes.grid': True,
              'lines.linewidth': 1,
              }
    import matplotlib
    if LooseVersion(matplotlib.__version__) >= '2':
        params['font.size'] = fontsize
    else:
        params['text.fontsize'] = fontsize
    plt.rcParams.update(params)

annot_kwargs = dict(fontsize=12, fontweight='bold',
                    xycoords="axes fraction", ha='right', va='center')

mask_params = dict(marker='*', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=15)