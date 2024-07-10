#import packages (can do pip install chemprop)
import pandas as pd
import numpy as np
from pathlib import Path
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
import csv

#DATA GENERATION

input_path = 'maml_newdata.csv' # path to your data .csv file
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
train_indices, val_indices, test_indices = data.make_split_indices(vp_mols, "random", (0.4, 0.4, 0.2))
train_data, val_data, test_data = data.split_data_by_indices(
    vp_data, train_indices, val_indices, test_indices
)
vp_train = data.MoleculeDataset(train_data, featurizer)
vp_val = data.MoleculeDataset(val_data, featurizer)
vp_test = data.MoleculeDataset(test_data, featurizer)


vd_mols = [d.mol for d in vd_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(vd_mols, "random", (0.4, 0.4, 0.2))
train_data, val_data, test_data = data.split_data_by_indices(
    vd_data, train_indices, val_indices, test_indices
)

vd_train = data.MoleculeDataset(train_data, featurizer)
vd_val = data.MoleculeDataset(val_data, featurizer)
vd_test = data.MoleculeDataset(test_data, featurizer)

tk_mols = [d.mol for d in tk_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(tk_mols, "random", (0.4, 0.4, 0.2))
train_data, val_data, test_data = data.split_data_by_indices(
    tk_data, train_indices, val_indices, test_indices
)

tk_train = data.MoleculeDataset(train_data, featurizer)
tk_val = data.MoleculeDataset(val_data, featurizer)
tk_test = data.MoleculeDataset(test_data, featurizer)

vp_train_loader = data.build_dataloader(vp_train, num_workers=num_workers)
vp_val_loader = data.build_dataloader(vp_val, num_workers=num_workers, shuffle=False)
vp_test_loader = data.build_dataloader(vp_test, num_workers=num_workers, shuffle=False)

vd_train_loader = data.build_dataloader(vd_train, num_workers=num_workers)
vd_val_loader = data.build_dataloader(vd_val, num_workers=num_workers, shuffle=False)
vd_test_loader = data.build_dataloader(vd_test, num_workers=num_workers, shuffle=False)

tk_train_loader = data.build_dataloader(tk_train, num_workers=num_workers)
tk_val_loader = data.build_dataloader(tk_val, num_workers=num_workers, shuffle=False)
tk_test_loader = data.build_dataloader(tk_test, num_workers=num_workers, shuffle=False)

mp = nn.BondMessagePassing(depth=3) # Make the mpnn
#agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
ffn = nn.RegressionFFN()

# I haven't experimented with this at all, not sure if it will affect the SSL
batch_norm = False

#initialize the model
#mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

fp_mpnn = models.MPNN.load_from_checkpoint('/home/bhanu/Documents/Chemprop_Models/default_fingerprint_model/checkpoints/best.ckpt')
fp_mpnn.to(torch.device("cpu"))
fp_mpnn.eval()


# Define the loss function and optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(ffn.parameters(), lr = 0.001)

def argforward(weights, inputs):
    output = F.linear(inputs,weights[0],weights[1])
    output = F.relu(output)
    output = F.linear(output,weights[2],weights[3])

    return output

# inner optimization loop
tasks = ['vd']
for epoch in range(500):
    total_loss = 0
    optimizer.zero_grad()
    for task in tasks:
        loader = None
        if task == 'vp':
            loader = vp_train_loader
            val_loader = vp_val_loader
        elif task =='vd':
            loader = vd_train_loader
            val_loader = vd_val_loader
        else:
            loader = tk_train_loader
            val_loader = tk_val_loader
        
        temp_weights=[w.clone() for w in ffn.parameters()]
        for batch in loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            fp = fp_mpnn.fingerprint(bmg)
            pred=argforward(temp_weights, fp)

            targets = targets.reshape(-1,1)
            loss = criterion(pred, targets)

            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-0.001*g for w,g in zip(temp_weights,grads)] #temporary update of weights

        for batch in val_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            fp = fp_mpnn.fingerprint(bmg)
            pred=argforward(temp_weights, fp)

            targets = targets.reshape(-1,1)
            metaloss = criterion(pred, targets)
            total_loss += metaloss


    metagrads=torch.autograd.grad(total_loss,ffn.parameters())
    #important step
    for w,g in zip(ffn.parameters(),metagrads):
        w.grad=g
    
    optimizer.step()
    print(total_loss)


final_preds = []
final_targets = []

for task in tasks:
    pred_out = []
    target_out = []
    temp_weights=[w.clone() for w in ffn.parameters()]
    loader = None
    if task == 'vp':
        loader = vp_train_loader
        test_loader = vp_test_loader
    elif task =='vd':
        loader = vd_train_loader
        test_loader = vd_test_loader
    else:
        loader = tk_train_loader
        test_loader = tk_test_loader
    
    for batch in loader:
        for i in range(5):
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            fp = fp_mpnn.fingerprint(bmg)
            pred=argforward(temp_weights, fp)

            targets = targets.reshape(-1,1)
            loss = criterion(pred, targets)
            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-0.001*g for w,g in zip(temp_weights,grads)] #temporary update of weights
    
    test_loss = 0
    for batch in test_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        fp = fp_mpnn.fingerprint(bmg)
        pred=argforward(temp_weights, fp)

        targets = targets.reshape(-1,1)
        test_loss += criterion(pred, targets)
        pred_out.extend(pred.detach().numpy())
        target_out.extend(targets.detach().numpy())

    print(task, test_loss)
    for i in range(len(pred_out)):
        pred_out[i] = pred_out[i][0]
    for i in range(len(target_out)):
        target_out[i] = target_out[i][0]

    final_preds.append(pred_out)
    final_targets.append(target_out)



#dict_vp = {'pred_vp':final_preds[0], 'target_vp':final_targets[0]}
dict_vd = {'pred_vd':final_preds[0], 'target_vd':final_targets[0]}
#dict_tk = {'pred_tk':final_preds[0], 'target_tk':final_targets[0]}

#vp_out = pd.DataFrame(dict_vp)
vd_out = pd.DataFrame(dict_vd)
#tk_out = pd.DataFrame(dict_tk)

#vp_out.to_csv('vp.csv')
vd_out.to_csv('vd.csv')
#tk_out.to_csv('tk.csv')
