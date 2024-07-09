#import packages (can do pip install chemprop)
import pandas as pd
import numpy as np
from pathlib import Path
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
import torch
import copy
import torch.optim as optim

#DATA GENERATION

input_path = 'maml_data.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe
num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading

smis = df['smiles']
targets = df['target']
task_labels = df['task']
tasks = ['vp', 'vd', 'tk']


task_smis = []
task_targets = []
for task in tasks:
    indices = task_labels == task
    task_smis.append(smis.loc[indices])
    task_targets.append(targets.loc[indices])

vp_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis[0], task_targets[0])]
vd_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis[1], task_targets[1])]
tk_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis[2], task_targets[2])]

#make the graphs and put into a dataset
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

vp_mols = [d.mol for d in vp_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(vp_mols, "random", (0.7, 0.2, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    vp_data, train_indices, val_indices, test_indices
)
vp_train = data.MoleculeDataset(train_data, featurizer)
vp_val = data.MoleculeDataset(val_data, featurizer)
vp_test = data.MoleculeDataset(test_data, featurizer)


vd_mols = [d.mol for d in vd_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(vd_mols, "random", (0.7, 0.2, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    vd_data, train_indices, val_indices, test_indices
)

vd_train = data.MoleculeDataset(train_data, featurizer)
vd_val = data.MoleculeDataset(val_data, featurizer)
vd_test = data.MoleculeDataset(test_data, featurizer)

tk_mols = [d.mol for d in tk_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(tk_mols, "random", (0.7, 0.2, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    tk_data, train_indices, val_indices, test_indices
)

tk_train = data.MoleculeDataset(train_data, featurizer)
tk_val = data.MoleculeDataset(val_data, featurizer)
tk_test = data.MoleculeDataset(test_data, featurizer)

vp_train_loader = data.build_dataloader(vp_train, num_workers=num_workers)
vp_val_loader = data.build_dataloader(vp_val, num_workers=num_workers, shuffle=False)
vp_test_loader = data.build_dataloader(vp_test, num_workers=num_workers, shuffle=False, batch_size = 1)

vd_train_loader = data.build_dataloader(vd_train, num_workers=num_workers)
vd_val_loader = data.build_dataloader(vd_val, num_workers=num_workers, shuffle=False)
vd_test_loader = data.build_dataloader(vd_test, num_workers=num_workers, shuffle=False, batch_size = 1)

tk_train_loader = data.build_dataloader(tk_train, num_workers=num_workers)
tk_val_loader = data.build_dataloader(tk_val, num_workers=num_workers, shuffle=False)
tk_test_loader = data.build_dataloader(tk_test, num_workers=num_workers, shuffle=False, batch_size = 1)

#mp = nn.BondMessagePassing(depth=3) # Make the mpnn
#agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
ffn = nn.RegressionFFN()

# I haven't experimented with this at all, not sure if it will affect the SSL
batch_norm = False

#initialize the model
#mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

fp_mpnn = models.MPNN.load_from_checkpoint('/Users/bhanumamillapalli/Desktop/Current_Work/2024_Research_Project/Models/sol-gnn/model_default/best.pt')
fp_mpnn.eval()

# Define the loss function and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = optim.SGD(ffn.parameters(), lr = 0.01)

# inner optimization loop
for epoch in range(100):
    total_loss = 0
    for task in tasks:
        optimizer.zero_grad()
        loader = None
        if task == 'vp':
            loader = vp_train_loader
        elif task =='vd':
            loader = vd_train_loader
        else:
            loader = tk_train_loader
        
        for batch in loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            fp = fp_mpnn(bmg)
            pred = ffn(fp)
            targets = targets.reshape(-1,1)
            loss = criterion(pred, targets)

            loss.backward()
            optimizer.step()
        
        total_loss += loss
        
    print(total_loss)


