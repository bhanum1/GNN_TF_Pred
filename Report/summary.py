import pandas as pd
import numpy as np
import scipy
import os

true_df = pd.read_csv('Data/cond/cond_test_data.csv')

truths = true_df['target']
T_labels = true_df['T']


datasets = ['cond_transfer_from_vd']
splits = ['0.1', '0.2']
#splits = ['0.1','0.2','0.3','0.4', '0.5','0.6','0.7']
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
            results_dict[folder][T][split] = {'SRCC':[], 'MAE':[]}

            MAE_vals, SRCC_vals =[],[]
            for i in range(10): #each model iteration
                model = 'm' + str(i)
                PATH_3 = PATH_2 + model + '.csv'
                df = pd.read_csv(PATH_3)

                lnA = df['lnA'][T_labels == int(T)]
                EaR = df['EaR'][T_labels == int(T)]
                temp = df['temperature'][T_labels == int(T)]

                preds = lnA + EaR * temp
                truths_T = truths[T_labels == int(T)]

                
                results_dict[folder][T][split]['MAE'].append(round(np.average(abs(truths_T - preds)),5))
                results_dict[folder][T][split]['SRCC'].append(round(scipy.stats.spearmanr(truths_T, preds)[0],5))

SM1,SM2,SM3,SM4,SM5 = [],[],[],[],[]
SS1,SS2,SS3,SS4,SS5 = [],[],[],[],[]
MM1,MM2,MM3,MM4,MM5 = [],[],[],[],[]
MS1,MS2,MS3,MS4,MS5 = [],[],[],[],[]

for folder in datasets:
    for split in splits:
        SM1.append(np.average(results_dict[folder]['1'][split]['SRCC']))
        SM2.append(np.average(results_dict[folder]['2'][split]['SRCC']))
        SM3.append(np.average(results_dict[folder]['3'][split]['SRCC']))
        SM4.append(np.average(results_dict[folder]['4'][split]['SRCC']))
        SM5.append(np.average(results_dict[folder]['5'][split]['SRCC']))
        
        SS1.append(np.std(results_dict[folder]['1'][split]['SRCC']))
        SS2.append(np.std(results_dict[folder]['2'][split]['SRCC']))
        SS3.append(np.std(results_dict[folder]['3'][split]['SRCC']))
        SS4.append(np.std(results_dict[folder]['4'][split]['SRCC']))
        SS5.append(np.std(results_dict[folder]['5'][split]['SRCC']))

        MM1.append(np.average(results_dict[folder]['1'][split]['MAE']))
        MM2.append(np.average(results_dict[folder]['2'][split]['MAE']))
        MM3.append(np.average(results_dict[folder]['3'][split]['MAE']))
        MM4.append(np.average(results_dict[folder]['4'][split]['MAE']))
        MM5.append(np.average(results_dict[folder]['5'][split]['MAE']))

        MS1.append(np.std(results_dict[folder]['1'][split]['MAE']))
        MS2.append(np.std(results_dict[folder]['2'][split]['MAE']))
        MS3.append(np.std(results_dict[folder]['3'][split]['MAE']))
        MS4.append(np.std(results_dict[folder]['4'][split]['MAE']))
        MS5.append(np.std(results_dict[folder]['5'][split]['MAE']))


    out_dict = {'Train Split':splits,
                'T1_SRCC_mean':SM1,
                'T2_SRCC_mean':SM2,
                'T3_SRCC_mean':SM3,
                'T4_SRCC_mean':SM4,
                'T5_SRCC_mean':SM5,
                'T1_SRCC_std':SS1,
                'T2_SRCC_std':SS2,
                'T3_SRCC_std':SS3,
                'T4_SRCC_std':SS4,
                'T5_SRCC_std':SS5,
                'T1_MAE_mean':MM1,
                'T2_MAE_mean':MM2,
                'T3_MAE_mean':MM3,
                'T4_MAE_mean':MM4,
                'T5_MAE_mean':MM5,
                'T1_MAE_std':MS1,
                'T2_MAE_std':MS2,
                'T3_MAE_std':MS3,
                'T4_MAE_std':MS4,
                'T5_MAE_std':MS5,
                }
    
    df = pd.DataFrame(out_dict)
    df.to_csv('results/summary/' + folder + '.csv')
