import os

# Paths 
DISK_PATH = "/run/media/clara_elk/Clara_Seagate/SoLaugh_AL/0_RAW_DATA/"
BIDS_PATH = "/home/claraelk/scratch/laughter_data/bids_data/"
RESULT_PATH = "/home/claraelk/scratch/laughter_data/results/"
PREPROC_PATH = "/home/claraelk/scratch/laughter_data/preproc/"
PERF_PATH = "/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/PERF/solaugh_task/results_active/"
BEHAV_PATH = RESULT_PATH + "discrimination/behavior"
FIG_PATH = RESULT_PATH + "meg/reports/figures/"

# Task parameters
SUBJ_LIST = [str(i).zfill(2) for i in range(1, 33)] 
SUBJ_LIST.remove('17') #S17 did not finish the task
SUBJ_CLEAN = SUBJ_LIST.remove('26')
SUBJ_CLEAN = SUBJ_LIST.remove('27')
SUBJ_CLEAN = SUBJ_LIST.remove('29')

RUN_LIST = [str(i).zfill(2) for i in range(1, 13)]
PASSIVE_RUN = RUN_LIST[1:5] # Select runs 02 to 05
ACTIVE_RUN = RUN_LIST[6:13] # Select runs 07 to 12
RS_RUN = ['01', '06'] # Resting state

EVENTS_ID = {'Laugh/Real' : 11, 'Laugh/Posed' : 12, 'Good' : 99, 'Miss' : 66, 
            'OffSet' : 5, 'Env/Real' : 21, 'Scra/Real' : 31, 'Env/Posed' : 22,
            'Scra/Posed' : 32, 'Start' : 68}

FREQS_LIST = [ [2, 4], [4, 8], [8, 12], [12, 30], [30, 120] ]
FREQS_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']