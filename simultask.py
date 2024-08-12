from chemprop import data, featurizers, models, nn
import torch
import torch.nn.functional as F
from data_processing import *
import torch.optim as optim
from BT_loss import BT_loss
from rdkit import RDLogger
from plots import plot_losses
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from chemprop.cli.utils.parsing import build_data_from_files, make_dataset
from pathlib import Path

batch_size = 64
num_workers = 0

epochs = 100

def make_loaders(datafile):
    df = pd.read_csv(datafile) #convert to dataframe

    smis = df['smiles']
    targets = df['target']
    temps = df['temperature']
    lnA_targets = df['lnA']

    all_data = [data.MoleculeDatapoint.from_smi(smi, y, temp=temp, lnA_target=lnA) for smi, y,temp,lnA in zip(smis, targets,temps,lnA_targets)]
    mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits

    grouped = df.groupby(df['splits'].str.lower())
    train_indices = grouped.groups.get("train", pd.Index([])).tolist()
    val_indices = grouped.groups.get("val", pd.Index([])).tolist()
    test_indices = grouped.groups.get("test", pd.Index([])).tolist()
    train_indices, val_indices, test_indices = [train_indices], [val_indices], [test_indices]

    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    train_data = train_data[0]
    val_data = val_data[0]
    test_data = test_data[0]

    train_dset = make_dataset(train_data, reaction_mode='REAC_PROD')
    val_dset = make_dataset(val_data,reaction_mode='REAC_PROD')
    test_dset = make_dataset(test_data, reaction_mode='REAC_PROD')

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle = True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = data.build_dataloader(test_dset, batch_size=len(test_data), num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader



dropout = 0
mp = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 256, activation = 'leakyrelu')
ffn = nn.RegressionFFN(dropout=dropout, input_dim = 256, hidden_dim = 256, n_layers = 1) # regression head
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
batch_norm = False

#initialize the model
mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric()])

#mpnn = mpnn.load_from_file('/home/bhanu/Documents/GitHub/Thermal_Fluid_Prediction_GNN/Barlow_Twins/BT_big.ckpt')
mpnn.to(device)
opt1 = optim.Adam(mpnn.parameters(), lr = 0.00001)

criterion = mpnn.predictor.criterion



best_val_loss_main = float('inf')
best_val_loss_simul = float('inf')

train_loader, val_loader, test_loader = make_loaders(datafile='test.csv')

for epoch in range(epochs):
    train_loss = 0
    for batch in train_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, temps, lnA_targets, EaR_targets = batch
        
        bmg.to(device)
        targets = targets.to(device).reshape(-1,1)
        mask = targets.isfinite().to(device)
        lnA_targets = lnA_targets.to(device)
        weights = weights.to(device)
        temps = temps.to(device)
        pred = mpnn(bmg).to(device)

        l = criterion(pred, targets, mask, weights, lt_mask, gt_mask, temps, lnA_targets, EaR_targets)

        train_loss += l.item()
        l.backward()
        opt1.step()
    
    val_loss = 0
    for batch in val_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, temps, lnA_targets, EaR_targets = batch

        bmg.to(device)
        targets = targets.to(device).reshape(-1,1)
        mask = targets.isfinite().to(device)
        lnA_targets = lnA_targets.to(device)
        weights = weights.to(device)
        temps = temps.to(device)
        pred = mpnn(bmg).to(device)

        l = criterion(pred, targets, mask, weights, lt_mask, gt_mask, temps, lnA_targets, EaR_targets)

        val_loss += l.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"hyper_parameters": mpnn.hparams, "state_dict": mpnn.state_dict()}, '/home/bhanu/Documents/temp_models/m' + str(model) + '.ckpt')
        


