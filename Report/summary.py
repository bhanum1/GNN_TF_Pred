import pandas as pd
import numpy as np
import scipy
import os

true_df = pd.read_csv('Data/visc/visc_test_data.csv')

truths = true_df['target']

datasets = ['visc']
splits = ['0.1','0.2','0.3','0.4']
#'0.5','0.6','0.7']

results_dict = dict()

PATH_0 = 'results/'
for folder in datasets: #each dataset
    PATH_1 = PATH_0 + folder + '/'
    results_dict[folder] = dict()

    for split in splits: #each train split
        PATH_2 = PATH_1 + split
        PATH_2 += '_preds/'
        results_dict[folder][split] = {'SRCC':[], 'MAE':[]}

        for i in range(10): #each model iteration
            PATH_3 = PATH_2 + 'm' + str(i) + '.csv'

            df = pd.read_csv(PATH_3)

            lnA = df['lnA']
            EaR = df['EaR']
            temp = df['temperature']

            preds = lnA + EaR * temp

            MAE = round(np.average(abs(truths - preds)),5)
            SRCC = round(scipy.stats.spearmanr(truths, preds)[0],5)

            results_dict[folder][split]['MAE'].append(MAE)
            results_dict[folder][split]['SRCC'].append(SRCC)

print(results_dict)

'''
for folder in datasets:
    working_dict = results_dict[folder]

    for 

'''
