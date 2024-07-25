import pandas as pd
import numpy as np
import scipy
import os

true_df = pd.read_csv('Data/visc/visc_test_data.csv')

truths = true_df['target']
T_labels = true_df['T']


datasets = ['visc']
splits = ['0.1','0.2','0.3','0.4', '0.5', '0.6']
T_index = ['1', '2', '3', '4', '5']
results_dict = dict()

PATH_0 = 'results/'
for folder in datasets: #each dataset
    PATH_1 = PATH_0 + folder + '/'
    results_dict[folder] = dict()

    for T in T_index: #each temperature
        results_dict[folder][T] = dict()

        for split in splits: #each train split
            PATH_2 = PATH_1 + split
            PATH_2 += '_preds/'
            results_dict[folder][T][split] = dict()

            for i in range(10): #each model iteration
                model = 'm' + str(i)
                PATH_3 = PATH_2 + model + '.csv'
                df = pd.read_csv(PATH_3)
                results_dict[folder][T][split][model] = {'MAE':[], 'SRCC':[]}

                
                lnA = df['lnA'][T_labels == T]
                EaR = df['EaR'][T_labels == T]
                temp = df['temperature'][T_labels == T]

                preds = lnA + EaR * temp

                MAE = round(np.average(abs(truths - preds)),5)
                SRCC = round(scipy.stats.spearmanr(truths, preds)[0],5)

                results_dict[folder][split][model]['MAE'].append(MAE)
                results_dict[folder][split]['SRCC'].append(SRCC)


print(results_dict)
'''
for folder in datasets:
    MAE_mean, SRCC_mean, MAE_dev, SRCC_dev = [],[],[],[]
    for split in splits:
        MAE_mean.append(np.average(results_dict[folder][split]['MAE']))
        SRCC_mean.append(np.average(results_dict[folder][split]['SRCC']))
        MAE_dev.append(np.std(results_dict[folder][split]['MAE']))
        SRCC_dev.append(np.std(results_dict[folder][split]['SRCC']))

    out_dict = {'Train Split':splits,
                'SRCC_average':SRCC_mean,
                'SRCC_stdev':SRCC_dev,
                'MAE_average':MAE_mean,
                'MAE_stdev':MAE_dev}
    
    df = pd.DataFrame(out_dict)
    df.to_csv('results/summary/' + folder + '.csv')
'''