import mne
import os
from mne_bids import BIDSPath, write_raw_bids
from src.params import DISK_PATH, BIDS_PATH, EVENTS_ID_BIDS
from src.params import ACTIVE_RUN, PASSIVE_RUN

# List of the directory for each participant
folder_subj = os.listdir(DISK_PATH) 

# Remove non subject files
folder_subj.remove('Cahier_manip_Laughter_MEG.pdf')
folder_subj.remove('S28.zip')
folder_subj.remove('S0_Pilote')
folder_subj.remove('S17_removed') #S17 did not finish task

# Set up to anonymize data
#anonymize = {daysback : 595, keep_his : False, keep_source : False}

# check if BIDS_PATH exists, if not, create it
if not os.path.isdir(BIDS_PATH):
    os.mkdir(BIDS_PATH)
    print("BIDS folder created at : {}".format(BIDS_PATH))
else:
    print("{} already exists.".format(BIDS_PATH))

# Go through all participants files (S1, S2...)
for idx, old_file in enumerate(folder_subj) : 

    old_path = os.path.join(DISK_PATH, old_file)
    file_list = os.listdir(old_path) 

    #Go through all run and noise files
    for CAfile in file_list :
   
        # For Task files
        if '_Laughter-' and '.ds' in CAfile and not '.zip' in CAfile and not 'NOISE' in CAfile and not 'procedure' in CAfile:

            subj = CAfile[2:4]
            run = CAfile[-5:-3]

            # Associate run with task
            if run == '01' or run == '06' :
                task = 'RS'
            elif run in PASSIVE_RUN :
                task = 'LaughterPassive'
            elif run in ACTIVE_RUN :
                task = 'LaughterActive'
            
            # Set up bids info
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
            date_raw = raw.info['meas_date'].strftime('%Y%m%d')

            # Add info for noise empty room
            noise_path = DISK_PATH + 'NOISE1Trial5min_noise_{}_01.ds'.format(date_raw)
            noise_raw_fname = os.path.join(old_path, CAfile)
            noise_raw = mne.io.read_raw_ctf(noise_raw_fname)  

            # Find events in file
            if task == 'LaughterActive' or task == 'LaughterPassive':
                events = mne.find_events(raw, initial_event=False)

                write_raw_bids(
                    raw, 
                    bids_path=laughter_bids_path,
                    events_data=events,
                    event_id=EVENTS_ID_BIDS,
                    empty_room=noise_raw,
                    overwrite=True
                    ) # Not optimal, will overwrite empty room every time
            else :
                write_raw_bids(raw, 
                               bids_path=laughter_bids_path, 
                               empty_room=noise_raw,
                               overwrite = True)
        
        # We don't want to convert all other NOISE files
        if 'NOISE' in CAfile and 'Trial' in CAfile :
            continue
        