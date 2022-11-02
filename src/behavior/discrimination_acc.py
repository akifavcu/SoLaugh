import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import data -> to put in params

path = '/run/media/claraek/Clara_Seagate/SoLaugh_MSc/0_RAW_DATA/'

all_run = [str(i).zfill(2) for i in range(7, 13)]
#all_run = ['07']

# subj_list = [str(i) for i in list(range(1,17))+[18,19,20,21,22,23,25,28,29,30,32]]
subj_list = [1, 2]
df_results = pd.DataFrame()

subj_loop = []
acc_list = []
hit_false = []
score = []
laugh_type = []
nb_resp = []

for subj in subj_list:

    for run in all_run:

        # Read raw file
        filename = 'S{}/CA0{}_Laughter-kjerbi_20161013_{}.ds'.format(subj, subj, run)
        fname = os.path.join(path, filename)
        raw = mne.io.read_raw_ctf(fname)

        # Find events
        events = mne.find_events(raw, initial_event=False)
        events_corrected = list(events[:, [2]])

        # Find miss response
        for i, miss in enumerate(events_corrected) :
            if miss == 5 and (events_corrected[i+1] == 11 or events_corrected[i+1] == 12): 
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