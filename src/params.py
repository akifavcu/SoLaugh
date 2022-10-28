# Paths 
DISK_PATH = '/run/media/claraek/Clara_Seagate/SoLaugh_AL/0_RAW_DATA/'
BIDS_PATH = '/run/media/claraek/Clara_Seagate/SoLaugh_MSc/BIDS_DATA/'

# Task parameters
SUBJECTS = [str(i).zfill(2) for i in range(0, 33)] # Need to remove participant 17
RUNS = [str(i).zfill(2) for i in range(1, 13)]
EVENTS_ID = {'LaughReal' : 11, 'LaughPosed' : 12, 'Hit' : 99, 'False' : 66, 'Offset' : 5}