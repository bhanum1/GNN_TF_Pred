import pandas as pd
import numpy as np
import scipy
import os

true_df = pd.read_csv('Data/visc/visc_test_data.csv')

truths = true_df['target']

results_dir = 'results'


for root, dirs, files in os.walk(results_dir):
    for filename in dirs:
        print(os.path.join(root, filename))
'''
for j in range(10): #make this iterate through all models
    df = pd.read_csv('results/visc/0.1_preds/m'+str(j) + '.csv')

    lnA = df['lnA']
    EaR = df['EaR']
    temp = df['temperature']

    preds = lnA + EaR * temp

    MAE = np.average(abs(truths - preds))
    SRCC = scipy.stats.spearmanr(truths, preds)[0]
'''
    
