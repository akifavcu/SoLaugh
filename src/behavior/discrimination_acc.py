import mne
import os
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from src.params import PERF_PATH, BEHAV_PATH
from statsmodels.stats.anova import AnovaRM


def all_perf_data(perf_path) :

    '''
   
    Merge all subject datasets and clean it
    
    '''

    df_data = pd.DataFrame()
    for subj_perf in os.listdir(perf_path) :
        if '.csv' in subj_perf : 
            df_subj = pd.read_csv(perf_path + subj_perf)
            df_data = pd.concat([df_data, df_subj])
    df_data['nb_resp'] = 1  

    # Export table into csv
    df_data.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive.csv', index = False)

    # Remove the response 'none'
    df_clean = df_data[df_data['response'].str.contains('none') == False]

    return df_data, df_clean

def compute_RT(df, df_performance) :

    ''' TODO '''

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
    df_save_RT = pd.DataFrame(list(dict_RT.items()))
    df_performance = pd.concat([df_performance, df_save_RT])

    return correct_RT, error_RT, lreal_RT, lposed_RT, df_performance

def compute_performance(df) :
    
    ''' TODO '''

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
    
    ''' 

    Apply one way repeated measured ANOVA to see :

    - accuracy differences between real and posed laughter
    - differences between correct and error
    
    Experimental design with repeated measures for 
    each participants for both types of laugh and 
    both types of response (rmANOVA or Wilcoxon test
    depending on the normality of the distribution)

    - Dependant variable : performance
    - Subject : all subjects
    - Within-subject factors : Laugh type and response type

    '''

    df_perf = df[['subID','active_laughType','response','nb_resp']].groupby(['subID','active_laughType','response'], as_index=False).count()
    print(df_perf.head())

    # Verification of the normality
    perf_count = df_perf['nb_resp']

    # QQplot
    fig = sm.qqplot(perf_count, line='45')
    plt.show()

    perf_count.hist()
    plt.show()

    # Compute normality 
    print(pg.normality(data = df_perf, dv = 'nb_resp', group = 'active_laughType'))
    print(pg.sphericity(data = df_perf, dv = 'nb_resp', subject = 'subID', within = 'active_laughType')[-1])

    # Apply statistical analysis (Wilcoxon test)
    real_laughter = df[df['active_laughType'].str.contains('real') == True]
    posed_laughter = df[df['active_laughType'].str.contains('posed') == True]

    # Perform wilcoxon test for non normalize data
    grouped_real = real_laughter[['subID','response','nb_resp']].groupby(['subID','response'], as_index=False).count()
    grouped_posed = posed_laughter[['subID','response','nb_resp']].groupby(['subID','response'], as_index=False).count()
    result_perf_wilcoxon = stats.wilcoxon(grouped_posed['nb_resp'], grouped_real['nb_resp'])
    print(result_perf_wilcoxon)

    # Perform ANOVA RM
    result_perf_anovarm = AnovaRM(df_perf, 'nb_resp', 'subID', within=['active_laughType','response'])
    result_perf = result_perf_anovarm.fit()
    print(result_perf)

    return result_perf_wilcoxon, result_perf_anovarm

def stat_RT(df) :
    
    ''' 

    Apply two way repeated measured ANOVA to see :

    - RT differences between correct-error responses
    - RT differences between real and posed laughter
    - Interaction between these two variables
    
    Experimental design with repeated measures for 
    each participants for both types of laugh and 
    both types of response (rmANOVA or Wilcoxon test
    depending on the normality of the distribution)

    - Dependant variable : reaction time
    - Subject : all subjects
    - Within-subject factors : Laugh type and response type

    '''

    df_RT = df[['subID','active_laughType','response','RT']].groupby(['subID','active_laughType','response'], as_index=False).mean()
    print(df_RT)

    # Verify if data is normally distributed
    # As our sample is > 50
    # we use Kolmogorov-Smirnov Test
    RT = df_RT['RT']
    print(pg.normality(data = df_RT, dv = 'RT', group = 'active_laughType'))
    print(pg.normality(data = df_RT, dv = 'RT', group = 'response'))

    fig = sm.qqplot(RT, line='45')
    plt.show()

    # Apply statistical analysis (Friedman test)
    result_stat_RT = pg.friedman(data=df_RT, dv="RT", within='response', subject="subID")
    # stats.friedmanchisquare
    print(result_stat_RT)

    # Perform ANOVA RM
    result_RT_anovarm = AnovaRM(df_RT, 'RT', 'subID', within=['active_laughType','response'])
    result_RT = result_perf_anovarm.fit()
    print(result_RT)

    return result_RT

def plot_perf(df) : 

    '''
    TODO : change figure for sns
    Plot behavioral results figures
    
    '''

    grouped_laugh = df.groupby(['subID', 'response', 'active_laughType']).count()
    print(grouped_laugh.head())

    grouped_RT = df.groupby(['subID', 'response', 'active_laughType']).mean()
    print(grouped_RT.head())

    # Plot differences between hit and false
    grouped_laugh.boxplot(column = 'nb_resp', by = ['response'])
    plt.ylabel('Number of answer')
    plt.title('Overall performance')
    plt.savefig(BEHAV_PATH + 'discrimination_performance-response.png')
    plt.show()

    sns.boxplot(data = df, x= 'active_laughType', y = 'nb_resp', palette = 'Set3')
    plt.show()

    # Plot differences between laughter type performance
    grouped_laugh.boxplot(column = 'nb_resp', by = ['response', 'active_laughType'])
    plt.ylabel('Number of answer')
    plt.title('Performance for each type of laughter')
    plt.savefig(BEHAV_PATH + 'discrimination_performance-real-posed.png')
    plt.show()

    sns.boxplot(data = df, x= 'active_laughType', y = 'nb_resp', hue = 'response', palette = 'Set3')
    plt.show()

    # Plot differences between RT for each laughter types
    grouped_RT.boxplot(column = 'RT', by = ['active_laughType'])
    plt.ylabel('RT (sec)')
    plt.title('RT for each type of laughter')
    plt.savefig(BEHAV_PATH + 'discrimination_RT-real-posed.png')
    plt.show()

    sns.boxplot(data = df, x= 'active_laughType', y = 'RT', palette = 'Set3') 
    plt.show()

    # Plot differences between RT for correct and error responses
    grouped_RT.boxplot(column = 'RT', by = ['response'])
    plt.ylabel('RT (sec)')
    plt.title('RT for each response category')
    plt.savefig(BEHAV_PATH + 'discrimination_RT-correct-error.png')
    plt.show()

    sns.boxplot(data = df, x= 'active_laughType', y = 'RT', hue = 'response', palette = 'Set3')
    plt.show()
    return grouped_laugh, grouped_RT


if __name__ == '__main__' :

    # check if the path exists, if not, create it
    if not os.path.isdir(BEHAV_PATH):
        os.mkdir(BEHAV_PATH)
        print('BEHAV folder created at : {}'.format(BEHAV_PATH))
    else:
        print('{} already exists.'.format(BEHAV_PATH))

    df_data, df_clean = all_perf_data(PERF_PATH)

    df_performance = compute_performance(df_clean)

    correct_RT, error_RT, lreal_RT, lposed_RT, df_performance = compute_RT(df_clean, df_performance)

    result_perf_wilcoxon, result_perf_anovarm = stat_perf(df_clean)
    result_stat_RT = stat_RT(df_clean)

    df_grouped_laugh, df_grouped_RT = plot_perf(df_clean)

    # Save final df performance with all info
    df_performance.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive_results-performance_RT_stats.csv', index = False)       
