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
input_path = 'maml_bigdata.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe
num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading

#get columns
smis = df['smiles']
targets = df['target']
task_labels = df['task']

#iterable for tasks
num_tasks = 10
tasks = range(num_tasks)

#create holders for all dataloaders
train_loaders = []
val_loaders = []
test_loaders = []
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

#separate data by task and put into loaders
for task in tasks:
    #get the smis and targets for the task
    indices = task_labels == task
    task_smis = smis.loc[indices]
    task_targets = targets.loc[indices]

    #create data
    data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis[task], task_targets[task])]
    mols = [d.mol for d in data]
    
    #split data
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.4, 0.4, 0.2)) #create split indices
    train_data, val_data, test_data = data.split_data_by_indices(
    data, train_indices, val_indices, test_indices
    )

    #create dataset
    train_dataset = data.MoleculeDataset(train_data, featurizer)
    val_dataset = data.MoleculeDataset(val_data, featurizer)
    test_dataset = data.MoleculeDataset(test_data, featurizer)

    #add to loaders
    train_loaders.append(data.build_dataloader(train_dataset, num_workers=num_workers,batch_size=1))
    val_loaders.append(data.build_dataloader(train_dataset, num_workers=num_workers,batch_size=1))
    test_loaders.append(data.build_dataloader(train_dataset, num_workers=num_workers,batch_size=1))

#mp = nn.BondMessagePassing(depth=3) # Make the mpnn
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
        train_loader = train_loaders[task]
        val_loader = val_loaders[task]
        test_loader = test_loaders[task]

        temp_weights=[w.clone() for w in ffn.parameters()]
        for batch in train_loader:
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

    train_loader = train_loaders[task]
    val_loader = val_loaders[task]
    test_loader = test_loaders[task]

    
    for batch in train_loader:
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
#dict_vd = {'pred_vd':final_preds[0], 'target_vd':final_targets[0]}
#dict_tk = {'pred_tk':final_preds[0], 'target_tk':final_targets[0]}

#vp_out = pd.DataFrame(dict_vp)
#vd_out = pd.DataFrame(dict_vd)
#tk_out = pd.DataFrame(dict_tk)

#vp_out.to_csv('vp.csv')
#vd_out.to_csv('vd.csv')
#tk_out.to_csv('tk.csv')
