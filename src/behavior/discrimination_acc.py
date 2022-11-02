import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.params import BIDS_PATH, SUBJ_LIST, ACTIVE_RUN, RESULT_PATH

# check if RESULT_PATH/behavior/behavior exists, if not, create it
BEHAV_PATH = RESULT_PATH + 'DISCRIMINATION/behavior/'

if not os.path.isdir(BEHAV_PATH):
    os.mkdir(BEHAV_PATH)
    print("BEHAV folder created at : {}".format(BEHAV_PATH))
else:
    print("{} already exists.".format(BEHAV_PATH))

# Initiate all list
df_results = pd.DataFrame()
subj_loop = []
acc_list = []
hit_false = []
score = []
laugh_type = []
nb_resp = []
laughter_predict = []

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
        # Compute reaction time

#Compute the number of response
for resp in hit_false :
    nb_resp.append(1)

# Implement a multi-index table
df = pd.DataFrame(data = {'Subject': subj_loop, 'Hit/False': hit_false, 'Score': score,
'LaughType': laugh_type, 'NumberResponse': nb_resp})

# Export table into csv
df.to_csv(RESULT_PATH + 'discrimination_performance.csv')

# Plot graph showing differences between hit and false
grouped_hit_false = df.groupby(['Subject', 'Hit/False', 'LaughType']).count()
grouped_hit_false.boxplot(column = 'NumberResponse', by = ['Hit/False'])
plt.ylabel('Number of answer')
plt.title('Overall performance')
plt.savefig(BEHAV_PATH + 'discrimination_performance-hit-false.png')
plt.show()

# Plot graph showing differences between laughter type performance
grouped_real_posed = df.groupby(['Subject', 'Hit/False', 'LaughType']).count()
grouped_real_posed.boxplot(column = 'NumberResponse', by = ['Hit/False', 'LaughType'])
plt.ylabel('Number of answer')
plt.title('Performance for each type of laughter')
plt.savefig(BEHAV_PATH + 'discrimination_performance-real-posed.png')
plt.show()

# Plot a confusion matrix
for i, resp in enumerate(df['Score']) :
    if resp == 0 :
        if df['LaughType'][i] == 'Real' :
            laughter_predict.append('Posed')
        elif df['LaughType'][i] == 'Posed' :
            laughter_predict.append('Real')
    elif resp == 1 :
        if df['LaughType'][i] == 'Real' :
            laughter_predict.append('Real')
        elif df['LaughType'][i] == 'Posed' :
            laughter_predict.append('Posed')

df['LaughPredict'] = laughter_predict

df['y_actual'] = df['LaughType'].map({'Real': 1, 'Posed': 0})
df['y_predicted'] = df['LaughPredict'].map({'Real': 1, 'Posed': 0})

confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], 
                    colnames=['Predicted'])

print(confusion_matrix)
sns.heatmap(confusion_matrix, annot=True)
plt.savefig(BEHAV_PATH + 'confusion_matrix_real-posed.png')
plt.show()
