# Paths 
DISK_PATH = '/run/media/clara_elk/Clara_Seagate/SoLaugh_AL/0_RAW_DATA/'
BIDS_PATH = '/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/bids_data/'
RESULT_PATH = '/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/results/'
PREPROC_PATH = '/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/preproc/'
PERF_PATH = '/run/media/clara_elk/Clara_Seagate/SoLaugh_MSc/PERF/solaugh_task/results_active/'
BEHAV_PATH = RESULT_PATH + 'discrimination/behavior/'

# Task parameters
SUBJ_LIST = [str(i).zfill(2) for i in range(1, 33)] 
SUBJ_LIST.remove('17') #S17 did not finish the task

RUN_LIST = [str(i).zfill(2) for i in range(1, 13)]
PASSIVE_RUN = RUN_LIST[1:5] # Select runs 02 to 05
ACTIVE_RUN = RUN_LIST[6:13] # Select runs 07 to 12
RS_RUN = ['01', '06'] # Resting state

EVENTS_ID = {'LaughReal' : 11, 'LaughPosed' : 12, 'Good' : 99, 'Bad' : 66, 
            'OffSet' : 5, 'EnvReal' : 21, 'ScraReal' : 31, 'EnvPosed' : 22,
            'ScraPosed' : 32, 'Start' : 68}