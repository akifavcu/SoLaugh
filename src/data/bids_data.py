import mne
import os
from mne_bids import BIDSPath, write_raw_bids
from src.params import DISK_PATH, BIDS_PATH, EVENTS_ID, RUNS
# Pb with the sys.path

# List of the directory for each participant
folder_subj = os.listdir(DISK_PATH) 
# Remove non subject files
folder_subj.remove('Cahier_manip_Laughter_MEG.pdf')
folder_subj.remove('S28.zip')
folder_subj.remove('S0_Pilote')
folder_subj.remove('S17_removed') #subject remove from the task 

passive_run = RUNS[1:5] # Select runs 02 to 05
active_run = RUNS[6:13] # Select runs 07 to 12

# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print("BIDS folder created at : {}".format(BIDS_PATH))
else:
    print("{} already exists.".format(BIDS_PATH))


for old_file in folder_subj : # Go through S1, S2 etc...

    old_path = os.path.join(DISK_PATH, old_file)
    file_list = os.listdir(old_path) 
    
    for CAfile in file_list : # Go through CA01_Laughter ...

        # For NOISE file 
        if 'NOISE_noise_' in CAfile and not '.fif' in CAfile:

            subj_number = old_file[1:] # Remove the letter S
            if len(subj_number) == 1 :
                subj = '0' + subj_number
            else :
                subj = str(subj_number)    

            noise_bids_path = BIDSPath(
                subject=subj,
                task='NOISE',
                session='noise',
                datatype='meg',
                extension='.ds', 
                root=BIDS_PATH
                )

            # Convert NOISE files into BIDS
            noise_raw_fname = os.path.join(old_path, CAfile)
            noise_raw = mne.io.read_raw_ctf(noise_raw_fname)
            write_raw_bids(noise_raw, bids_path=noise_bids_path, overwrite = True)
   
        # For Task file
        if '_Laughter-' and '.ds' in CAfile and not '.zip' in CAfile and not 'NOISE' in CAfile and not 'procedure' in CAfile:

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
                task=task,
                session='recording',
                datatype='meg',
                extension='.ds', 
                root=BIDS_PATH
                ) 
                
            # Convert laughter files into BIDS 
            raw_fname = os.path.join(old_path, CAfile)
            raw = mne.io.read_raw_ctf(raw_fname)

            if task == 'LaughterActive' or task == 'LaughterPassive':
                events = mne.find_events(raw, initial_event=False)

                write_raw_bids(
                    raw, 
                    bids_path=laughter_bids_path,
                    events_data=events,
                    event_id=EVENTS_ID,
                    overwrite=True
                    )
            else :
                write_raw_bids(raw, bids_path=laughter_bids_path, overwrite = True)
        
        if 'NOISE' in CAfile and 'Trial' in CAfile :
            continue
        