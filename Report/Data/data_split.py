import pandas as pd
import numpy as np

df = pd.read_csv('hc/hydrocarbon_visc.csv')


train_order = df['Train_order'].dropna()

smiles = df['smiles']
temperature = df['temp_input']
lnA = df['lnA_input']
target = df['target']
splits = df['splits']



train_fracs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]


for frac in train_fracs:
    new_df = pd.DataFrame(columns=['smiles','temperature','lnA','target', 'splits'])
    num_train = int(np.floor(frac * len(smiles)/5))

    for col in train_order[:num_train]:
        slice = np.where(splits == str(int(col)))
        
        for i in np.nditer(slice):
            row_to_append = pd.DataFrame([{'smiles':smiles[i], 
                                        'temperature':temperature[i],
                                        'lnA':lnA[i],
                                        'target':target[i],
                                        'splits':'train'}])
            
            new_df = pd.concat([new_df,row_to_append])

    val_slice = np.where(splits == 'val')
    for i in np.nditer(val_slice):
            row_to_append = pd.DataFrame([{'smiles':smiles[i], 
                                        'temperature':temperature[i],
                                        'lnA':lnA[i],
                                        'target':target[i],
                                        'splits':'val'}])
            
            new_df = pd.concat([new_df,row_to_append])

    test_slice = np.where(splits == 'test')
    for i in np.nditer(test_slice):
            row_to_append = pd.DataFrame([{'smiles':smiles[i], 
                                        'temperature':temperature[i],
                                        'lnA':lnA[i],
                                        'target':target[i],
                                        'splits':'test'}])
            
            new_df = pd.concat([new_df,row_to_append])

    new_df.to_csv('hc/hc_' + str(frac) + '.csv')