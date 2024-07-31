from chemprop import data, featurizers, models, nn
import torch
import torch.nn.functional as F
from data_processing import *
import torch.optim as optim
from BT_loss import BT_loss
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
#Create the network
dropout = 0
mp_1 = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 3) # Make the gnn
mp_2 = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 3) # Make the gnn
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.

opt1 = optim.Adam(mp_1.parameters(), lr = 0.0001)
opt2 = optim.Adam(mp_2.parameters(), lr = 0.0001)

from chemprop.cli.utils.parsing import build_data_from_files, make_dataset
from pathlib import Path

input_path = 'Barlow_Twins/dataset.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe

smis = df['smiles']

epochs = 100
for epoch in range(epochs):
    train, val, train_2, val_2 = create_loaders(smis, batch_size=64, num_workers=10)

    train_l = 0
    for batch_1, batch_2 in zip(train, train_2):
        bmg_1, V_d_1, X_d, targets, weights, lt_mask, gt_mask = batch_1
        bmg_2, V_d_2, X_d, targets, weights, lt_mask, gt_mask = batch_2
        
        r_1 = mp_1.forward(bmg_1,V_d_1)
        r_1 = agg.forward(r_1, bmg_1.batch)
        r_2 = mp_2.forward(bmg_2,V_d_2)
        r_2 = agg.forward(r_2, bmg_2.batch)

        _, _, loss = BT_loss(r_1, r_2, [1, 0.01])
        train_l += loss

    train_l.backward()
    opt1.step()
    opt2.step()

    val_loss = 0
    for batch_1, batch_2 in zip(val, val_2):

        bmg_1, V_d_1, X_d, targets, weights, lt_mask, gt_mask = batch_1
        bmg_2, V_d_2, X_d, targets, weights, lt_mask, gt_mask = batch_2
        
        r_1 = mp_1.forward(bmg_1,V_d_1)
        r_1 = agg.forward(r_1, bmg_1.batch)
        r_2 = mp_2.forward(bmg_2,V_d_2)
        r_2 = agg.forward(r_2, bmg_2.batch)

        _, _, loss = BT_loss(r_1, r_2, [1, 0.01])

        val_loss += loss
    
    val_size = np.floor(len(smis) * 0.1) / len(bmg_1)
    print("Train Loss: {0:.2f} Val Loss: {1:.2f}".format(train_l / (val_size * 9), val_loss / val_size))