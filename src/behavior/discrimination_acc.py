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

def compute_performance_and_RT(df) :
    
    ''' TODO '''

    df_bool = df['response'].str.contains('correct')
    correct_count = []

    for idx, resp in enumerate(df_bool) :
        if df_bool.iloc[idx] == True:
            correct_count.append(1)
        else :
            correct_count.append(0)

    df['correct_count'] = correct_count

    df_real_laughter = df[df['active_laughType'].str.contains('real') == True]
    df_posed_laughter = df[df['active_laughType'].str.contains('posed') == True]
    
    # Compute the number and the % of correct answer in total
    sum_all = df['correct_count'].sum()
    print('Overall nb of correct answer:', sum_all)

    # Compute the % of correct answers in total
    perc_all = (sum_all*100)/len(df)

    print('Overall percentage of correct answers:', perc_all, '\n')

    # Compute the number of correct answer for each laughter
    sum_real = df_real_laughter['correct_count'].sum()
    sum_posed = df_posed_laughter['correct_count'].sum()
    print('Nb of correct answer for real laughter:', sum_real)
    print('Nb of correct answer for posed laughter:', sum_posed, '\n')

    # Compute the % of correct answers for each laughter
    perc_real = (sum_real*100)/len(df_real_laughter)
    perc_posed = (sum_posed*100)/len(df_posed_laughter)
    print('Percentage of correct answers for real laughter:', perc_real)
    print('Percentage of correct answers for posed laughter:', perc_posed, '\n')

    # Save results in csv files
    dict_perf = {'nb_correct_all' : sum_all, 
                'nb_correct_real' : sum_real, 
                'nb_correct_posed' : sum_posed, 
                'percentage_correct_all' : perc_all,
                'percentage_correct_real' : perc_real,
                'percentage_correct_posed' : perc_posed}

    df_performance = pd.DataFrame(list(dict_perf.items()), columns = ['performance', 'values'])      

    # Compute mean and STD of reaction times
    mean_RT = np.mean(df['RT'])
    std_RT = stats.tstd(df['RT'])
    print('Moyenne RT :', mean_RT)
    print('STD RT :', std_RT)

    correct_df = df[df['response'].str.contains('correct') == True]
    error_df = df[df['response'].str.contains('error') == True]

    correct_mean_RT = np.mean(correct_df['RT'])
    correct_std_RT = stats.tstd(correct_df['RT'])
    print('Moyenne RT correct:', correct_mean_RT)
    print('STD RT correct:', correct_std_RT)
    
    error_mean_RT = np.mean(error_df['RT'])
    error_std_RT = stats.tstd(error_df['RT'])
    print('Moyenne RT error:', error_mean_RT)
    print('STD RT error:', error_std_RT)

    # Take RT for both laughter
    real_RT = df_real_laughter['RT']
    posed_RT = df_posed_laughter['RT']
    
    real_mean_RT = np.mean(df_real_laughter['RT'])
    real_std_RT = stats.tstd(df_real_laughter['RT'])
    print('Moyenne RT real laughter:', mean_RT)
    print('STD RT real laughter:', std_RT)
    
    posed_mean_RT = np.mean(df_posed_laughter['RT'])
    posed_std_RT = stats.tstd(df_posed_laughter['RT'])
    print('Moyenne RT posed laughter:', mean_RT)
    print('STD RT posed laughter :', std_RT)

    # Save Mean and STD in a csv file
    dict_RT = {'mean_RT' : mean_RT,
                'std_RT' : std_RT,
                'correct_mean_RT' : correct_mean_RT,
                'correct_std_RT' : correct_std_RT,
                'error_mean_RT' : error_mean_RT,
                'error_std_RT' : error_std_RT,
                'real_mean_RT' : real_mean_RT,
                'real_std_RT' : real_std_RT,
                'posed_mean_RT' : posed_mean_RT,
                'posed_std_RT' : posed_std_RT}

    df_save_RT = pd.DataFrame(list(dict_RT.items()), columns = ['performance', 'values'])
    df_performance = pd.concat([df_performance, df_save_RT])

    return df_performance

def stat_performance(df) :
    
    ''' 

    Apply one way repeated measured ANOVA to see :

    - accuracy differences between real and posed laughter
    - differences between correct and error
    
    Experimental design with repeated measures for 
    each participants for both types of laugh and 
    both types of response (rmANOVA)

    - Dependant variable : performance
    - Subject : all subjects
    - Within-subject factors : Laugh type and response type

    '''

    df_perf = df[['subID','active_laughType','response','nb_resp']].groupby(['subID','active_laughType','response'], as_index=False).count()

    # Verification of the normality
    perf_count = df_perf['nb_resp']
    print('\nNormality test for perf-laugh:\n', pg.normality(data = df_perf, dv = 'nb_resp', group = 'active_laughType'))
    print('Normality test for perf-response:\n', pg.normality(data = df_perf, dv = 'nb_resp', group = 'response'))

    #Verification of the sphericity
    spher, _, chisq, dof, pval = pg.sphericity(df_perf, dv='nb_resp',
                                           subject='subID',
                                           within=['active_laughType', 'response'])
    print('Test sphericity nb_resp:', spher, round(chisq, 3), dof, round(pval, 3), '\n')

    # Perform ANOVA RM
    result_perf_anovarm = pg.rm_anova(data=df_perf, dv='nb_resp', subject='subID', 
                                    within=['active_laughType','response'],
                                    correction='auto', detailed=True)
    print('Result ANOVARM for performance:\n', result_perf_anovarm)

    # Save perf stats results in csv file
    result_perf_anovarm.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive_res-perf-ANOVArm.csv', index = False)

    return result_perf_anovarm

def stat_RT(df) :
    
    ''' 

    Apply two way repeated measured ANOVA to see :

    - RT differences between correct-error responses
    - RT differences between real and posed laughter
    - Interaction between these two variables
    
    Experimental design with repeated measures for 
    each participants for both types of laugh and 
    both types of response (rmANOVA)

    - Dependant variable : reaction time
    - Subject : all subjects
    - Within-subject factors : Laugh type and response type

    '''

    df_RT = df[['subID','active_laughType','response','RT']].groupby(['subID','active_laughType','response'], as_index=False).mean()

    # Verify if data is normally distributed
    RT = df_RT['RT']
    print('\nNormality test for RT-laugh:\n', pg.normality(data = df_RT, dv = 'RT', group = 'active_laughType'))
    print('Normality test for RT-response:\n', pg.normality(data = df_RT, dv = 'RT', group = 'response'))

    #Verification of the sphericity
    spher, _, chisq, dof, pval = pg.sphericity(df_RT, dv='RT',
                                           subject='subID',
                                           within=['active_laughType', 'response'])
    print('Test sphericity RT:', spher, round(chisq, 3), dof, round(pval, 3), '\n')

    # Perform ANOVA RM
    result_RT_anovarm = pg.rm_anova(data=df_RT, dv='RT', subject='subID', 
                                    within=['active_laughType','response'],
                                    correction='auto', detailed=True)
    print('\nResult ANOVARM for RT:\n', result_RT_anovarm)

    # Save stats results
    result_RT_anovarm.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive_res-RT-ANOVArm.csv', index = False)

    return result_RT_anovarm


if __name__ == '__main__' :

    # check if the path exists, if not, create it
    if not os.path.isdir(BEHAV_PATH):
        os.mkdir(BEHAV_PATH)
        print('BEHAV folder created at : {}'.format(BEHAV_PATH))
    else:
        print('{} already exists.'.format(BEHAV_PATH))

    df_data, df_clean = all_perf_data(PERF_PATH)

    df_performance = compute_performance_and_RT(df_clean)

    result_perf_anovarm = stat_performance(df_clean)
    result_RT_anovarm = stat_RT(df_clean)

    plot_perf(df_clean)

    # Save final df performance with all info
    df_performance.to_csv(BEHAV_PATH + 'subj-all_task-LaughterActive_res-performance-RT.csv', index = False)       
