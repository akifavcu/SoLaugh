import mne
import os
from mne_bids import BIDSPath, write_raw_bids
from params import DISK_PATH, BIDS_PATH, SUBJECTS, RUNS


# List of the directory for each participant
folder_data = os.listdir(DISK_PATH) 
# Remove non subject file
folder_data.remove('Cahier_manip_Laughter_MEG.pdf')
folder_data.remove('S28.zip')


# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print("BIDS folder created at : {}".format(BIDS_PATH))
else:
    print("{} already exists.".format(BIDS_PATH))


for old_file in folder_data :

    old_path = os.path.join(disk_path, folder_data)
    file_list = os.listdir(old_path)
    
    for subj in SUBJECTS:
        for run in RUNS:
    
    # For NOISE file        
    for files in file_list :
        if 'NOISE' in files :
    
            noise_bids_path = BIDSPath(
                subject=subj,
                session='01' 
                run=run,
                task='NOISE'
                datatype='meg',
                extension='.ds' 
                root=BIDS_PATH)

            write_raw_bids(raw, bids_path=noise_bids_path)
    
    # For Task file

