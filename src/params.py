# importations
import os

# Paths 
DISK_PATH = "/home/claraelk/scratch/laughter_data/0_RAW_DATA/"
BIDS_PATH = "/home/claraelk/scratch/laughter_data/laugh_bids_data/"
PREPROC_PATH = "/home/claraelk/scratch/laughter_data/preproc_bids_data/"
RESULT_PATH = "/home/claraelk/scratch/laughter_data/results/"
PERF_PATH = "/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/PERF/solaugh_task/results_active/"
MRI_SCAN = "/home/claraelk/scratch/laughter_data/results/mri/anat/subjects"
MRI_PATH = "/home/claraelk/scratch/laughter_data/results/mri/"
BEHAV_PATH = RESULT_PATH + "discrimination/behavior"
FIG_PATH = RESULT_PATH + "meg/reports/figures/"

# Task parameters
SUBJ_LIST = [str(i).zfill(2) for i in range(1, 33)] 
SUBJ_LIST.remove('17') #S17 did not finish the task
SUBJ_LIST.remove('26')
SUBJ_LIST.remove('27')
SUBJ_LIST.remove('29')
SUBJ_LIST.remove('31')
SUBJ_CLEAN = SUBJ_LIST

RUN_LIST = [str(i).zfill(2) for i in range(1, 13)]
PASSIVE_RUN = RUN_LIST[1:5] # Select runs 02 to 05
ACTIVE_RUN = RUN_LIST[6:13] # Select runs 07 to 12
RS_RUN = ['01', '06'] # Resting state

EVENTS_ID_BIDS = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Bad' : 66, 
            'OffSet' : 5, 'EnvReal' : 21, 'ScraReal' : 31, 'EnvPosed' : 22,
            'ScraPosed' : 32, 'Start' : 68}

EVENTS_ID = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Miss' : 66, 
            'OffSet' : 5, 'EnvReal' : 21, 'ScraReal' : 31, 'EnvPosed' : 22,
            'ScraPosed' : 32, 'Start' : 68}

FREQS_LIST = [ [2, 4], [4, 8], [8, 12], [12, 30], [30, 120] ]
FREQS_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
