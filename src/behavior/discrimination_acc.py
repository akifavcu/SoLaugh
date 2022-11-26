import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from src.params import PERF_PATH, BEHAV_PATH

def all_perf_data(perf_path) :

    '''TODO'''

    df_data = pd.DataFrame()
    for subj_perf in os.listdir(perf_path) :
        if '.csv' in subj_perf : 
            df_subj = pd.read_csv(perf_path + subj_perf)
            df_data = pd.concat([df_data, df_subj])
    df_data['nb_resp'] = 1  

    # Export table into csv
    df_data.to_csv(BEHAV_PATH + 'discrimination_performance.csv', index = False)
    return df_data

def plot_perf(df) : 

    '''TODO'''

    # Remove the response 'none'
    df_clean = df[df['response'].str.contains('none') == False]

    # Plot differences between hit and false
    grouped = df_clean.groupby(['subID', 'response', 'active_laughType']).count()
    grouped.boxplot(column = 'nb_resp', by = ['response'])
    plt.ylabel('Number of answer')
    plt.title('Overall performance')
    plt.savefig(BEHAV_PATH + 'discrimination_performance-response.png')
    #plt.show()

    # Plot differences between laughter type performance
    grouped.boxplot(column = 'nb_resp', by = ['response', 'active_laughType'])
    plt.ylabel('Number of answer')
    plt.title('Performance for each type of laughter')
    plt.savefig(BEHAV_PATH + 'discrimination_performance-real-posed.png')
    #plt.show()

    return grouped, df_clean

def performance(df) :
    real_laughter = df[df['active_laughType'].str.contains('real') == True]
    posed_laughter = df[df['active_laughType'].str.contains('posed') == True]
    
    correct_resp = 0
    for resp in df['response'] :
        if resp == 'correct' :
            correct_resp += 1

    correct_acc = (correct_resp*100)/len(df)
    print(correct_acc)

    posed_real_acc = 0 

    return posed_real_acc, correct_acc

if __name__ == '__main__' :

    # check if the path exists, if not, create it
    if not os.path.isdir(BEHAV_PATH):
        os.mkdir(BEHAV_PATH)
        print('BEHAV folder created at : {}'.format(BEHAV_PATH))
    else:
        print('{} already exists.'.format(BEHAV_PATH))

    df_data = all_perf_data(PERF_PATH)

    df_grouped, df_clean = plot_perf(df_data)

    posed_real_acc, correct_acc = performance(df_clean)


