import mne
import os
from mne_bids import BIDSPath, write_raw_bids
from params import DISK_PATH, BIDS_PATH, EVENTS_ID


# List of the directory for each participant
folder_subj = os.listdir(DISK_PATH) 
# Remove non subject file
folder_subj.remove('Cahier_manip_Laughter_MEG.pdf')
folder_subj.remove('S28.zip')

passive_run = RUNS[1:6] # Select runs 02 to 05
active_run = RUNS[7:13] # Select runs 07 to 12

# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print("BIDS folder created at : {}".format(BIDS_PATH))
else:
    print("{} already exists.".format(BIDS_PATH))


for old_file in folder_subj : # Go through S1, S2 etc...

    old_path = os.path.join(disk_path, old_file)
    file_list = os.listdir(old_path) 
    
    # For NOISE file        
    for CAfile in file_list : # Go through CA01_Laughter ...
        subj_number = folder_subj[1:] # Remove the letter S
        if len(subj) == 1 :
            subj = '0' + subj_number

        if 'NOISE_noise_' in CAfile :
            noise_bids_path = BIDSPath(
                subject=subj,
                task='NOISE'
                datatype='meg',
                extension='.ds' 
                root=BIDS_PATH
                )

            # Convert NOISE files into BIDS
            raw_fname = os.path.join(oldpath, CAfile)
            raw = mne.io.read.cft(raw_fname)
            write_raw_bids(raw, bids_path=noise_bids_path, overwrite = True)
   
        # For Task file
        if 'Laughter' in CAfile :

            subj = CAfile[2:4]
            run = CAfile[-5:-3]

            if run == '01' or run == '06' :
                task = 'RS'
            elif run in passive_run :
                task = 'LaughterPassive'
            elif run in active_run :
                task = 'LaughterActive'
            
            laughter_bids_path = BIDSPath(                
                subject=subj, 
                run = run,
                task=task
                datatype='meg',
                extension='.ds' 
                root=BIDS_PATH
                ) 
                
            # Convert laughter files into BIDS 
            raw_fname = os.path.join(oldpath, CAfile)
            raw = mne.io.read.cft(raw_fname)

            if task == 'LaughterActive' or task == 'LaughterPassive':
                events = mne.find_events(raw)

                write_raw_bids(
                    raw, 
                    bids_path=laughter_bids_path, 
                    events_data = events,
                    event_id = EVENTS_ID,
                    overwrite = True
                    )
            else :
                write_raw_bids(raw, bids_path=laughter_bids_path)

        # For procedure file
        if 'procedure' in CAfile : 
            subj = CAfile[2:4]
            
            procedure_bids_path = BIDSPath(                
                subject=subj,
                task='procedure'
                datatype='meg',
                extension='.ds' 
                root=BIDS_PATH
                ) 

            # Convert procedure file into BIDS
            raw_fname = os.path.join(oldpath, CAfile)
            raw = mne.io.read.cft(raw_fname)  
            write_raw_bids(raw, bids_path=procedure_bids_path, overwrite = True)

