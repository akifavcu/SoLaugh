import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.params import BIDS_PATH, SUBJ_LIST, ACTIVE_RUN

df_results = pd.DataFrame()

subj_loop = []
acc_list = []
hit_false = []
score = []
laugh_type = []
nb_resp = []

for subj in SUBJ_LIST:

    for run in ACTIVE_RUN:

        # Read raw file
        filename = 'sub-{}/ses-recording/meg/sub-{}_ses-recording_task-LaughterActive_run-{}_meg.ds'.format(subj, subj, run)
        fname = os.path.join(BIDS_PATH, filename)
        raw = mne.io.read_raw_ctf(fname)

        # Find events
        events = mne.find_events(raw, initial_event=False)
        events_corrected = list(events[:, [2]])

        # Interpolate missing answers
        for i, ev in enumerate(events_corrected) :
            if i == (len(events_corrected)-1) and ev == 5 :
                events_corrected.insert(i+1, [66])
            elif ev == 5 and (events_corrected[i+1] == 11 or events_corrected[i+1] == 12): 
                # Add False where response is missing
                events_corrected.insert(i+1, [66])

        events_array = np.array(events_corrected)

        # Compute real and posed laugh
        for stim in events_array :
            if stim == 11:
                laugh_type.append('Real')
            elif stim == 12 :
                laugh_type.append('Posed')

        # Compute Hit and False answers
        for ev in events_array :
            if ev == 99 :
                hit_false.append('Hit')
                subj_loop.append(subj)
                score.append(1)
                
            elif ev == 66 :
                hit_false.append('False')
                subj_loop.append(subj)
                score.append(0)

        # TODO : compute reaction time -> events[:, [0]]

#Compute the number of response
for resp in hit_false :
    nb_resp.append(1)

# Implement a multi-index table
df = pd.DataFrame(data = {'Subject': subj_loop, 'Hit/False': hit_false, 'Score': score,
'LaughType': laugh_type, 'NumberResponse': nb_resp})
print(df.groupby(['Subject', 'Hit/False', 'LaughType']).count())

# TODO : Export table into csv

# Plot a graph
# TODO : Try to find a way to do graph with seaborn

#sns.boxplot(data = df, x = 'Hit/False', y = 'NumberResponse') 
#plt.show()

grouped = df.groupby(['Subject', 'Hit/False', 'LaughType']).count()
grouped.boxplot(column = 'NumberResponse', by = ['Hit/False'])
plt.show()

# TODO : Save figures