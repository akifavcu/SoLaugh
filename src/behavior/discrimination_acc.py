import mne
import os
import scipy
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
    df_data.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive.csv', index = False)
    return df_data

def compute_RT(df, df_performance) :
    # Need to compute reaction time
    # Take reaction time
    # Take RT for correct and wrong answers
    mean_RT = np.mean(df['RT'])
    std_RT = stats.tstd(df['RT'])
    print('Moyenne RT :', mean_RT)
    print('STD RT :', std_RT)

    correct_df = df[df['response'].str.contains('correct') == True]
    correct_RT = correct_df['RT']

    error_df = df[df['response'].str.contains('error') == True]
    error_RT = error_df['RT']

    # Take RT for both laughter
    lreal_df = df[df['active_laughType'].str.contains('real') == True]
    lreal_RT = lreal_df['RT']

    lposed_df = df[df['active_laughType'].str.contains('posed') == True]
    lposed_RT = lposed_df['RT']

    # TODO : Save mean and STD RT 
    dict_RT = {'mean_RT' : mean_RT,
                'std_RT' : std_RT}
    df_save_RT = pd.DataFrame(list(dict_RT))
    df_performance = pd.concat([df_performance, df_save_RT])

    return correct_RT, error_RT, lreal_RT, lposed_RT, df_performance

def compute_performance(df) :
    
    df_bool = df['response'].str.contains('correct')
    correct_count = []

    for idx, resp in enumerate(df_bool) :
        if df_bool.iloc[idx] == True:
            correct_count.append(1)
        else :
            correct_count.append(0)

    df['correct_count'] = correct_count

    real_laughter = df[df['active_laughType'].str.contains('real') == True]
    posed_laughter = df[df['active_laughType'].str.contains('posed') == True]
    
    # Compute the number and the % of correct answer in total
    sum_all = df['correct_count'].sum()
    print('Nb of correct answer in total :', sum_all)

    # Compute the % of correct answers in total
    perc_all = (sum_all*100)/len(df)

    print('Percentage of correct answers in total :', perc_all)

    # Compute the number of correct answer for each laughter
    sum_real = real_laughter['correct_count'].sum()
    sum_posed = posed_laughter['correct_count'].sum()
    print('Nb of correct answer for real laughter :', sum_real)
    print('Nb of correct answer for posed laughter :', sum_posed)

    # Compute the % of correct answers for each laughter
    perc_real = (sum_real*100)/len(real_laughter)
    perc_posed = (sum_posed*100)/len(posed_laughter)
    print('Percentage of correct answers for real laughter :', perc_real)
    print('Percentage of correct answers for posed laughter :', perc_posed)

    # Save results in csv files
    dict_perf = {'nb_correct_all' : sum_all, 
                'nb_correct_real' : sum_real, 
                'nb_correct_posed' : sum_posed, 
                'percentage_correct_all' : perc_all,
                'percentage_correct_real' : perc_real,
                'percentage_correct_posed' : perc_posed}

    df_performance = pd.DataFrame(list(dict_perf.items()), columns = ['performance', 'values'])      
    return df_performance

def stat_perf(df) :
    # Need to do stats for the performance
    # T-test
    # p-val ?
    real_laughter = df[df['active_laughType'].str.contains('real') == True]
    posed_laughter = df[df['active_laughType'].str.contains('posed') == True]


def stat_RT(correct_RT, error_RT, lreal_RT, lposed_RT) :
    
    print(stats.shapiro(correct_RT))
    print(stats.shapiro(error_RT))
    stat, p = stats.levene(correct_RT, error_RT)
    print(stat, p)

    stats_RT_correct_error = stats.ttest_ind(correct_RT, error_RT, )
    print(stats_RT_correct_error)

    stats_RT_laughter = stats.ttest_ind(lreal_RT, lposed_RT)
    print(stats_RT_laughter)

    return stats_RT_correct_error, stats_RT_laughter

# TODO : Move this into visualization
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

    # TODO : plot RT
    return grouped, df_clean


if __name__ == '__main__' :

    # check if the path exists, if not, create it
    if not os.path.isdir(BEHAV_PATH):
        os.mkdir(BEHAV_PATH)
        print('BEHAV folder created at : {}'.format(BEHAV_PATH))
    else:
        print('{} already exists.'.format(BEHAV_PATH))

    df_data = all_perf_data(PERF_PATH)

    df_grouped, df_clean = plot_perf(df_data)

    df_performance = compute_performance(df_clean)

    correct_RT, error_RT, lreal_RT, lposed_RT, df_performance = compute_RT(df_data, df_performance)

    stats_RT_correct_error, stats_RT_laughter = stat_RT(correct_RT, error_RT, lreal_RT, lposed_RT)

    # Save final df performance with all info
    df_performance.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive_performance_correct.csv', index = False)       
