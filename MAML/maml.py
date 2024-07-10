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
from torch import nn as tnn

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
    task_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis, task_targets)]
    mols = [d.mol for d in task_data]
    
    
    #split data
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.4, 0.4, 0.2)) #create split indices
    train_data, val_data, test_data = data.split_data_by_indices(
    task_data, train_indices, val_indices, test_indices
    )

    #create dataset
    train_dataset = data.MoleculeDataset(train_data, featurizer)
    val_dataset = data.MoleculeDataset(val_data, featurizer)
    test_dataset = data.MoleculeDataset(test_data, featurizer)

    #add to loaders
    train_loaders.append(data.build_dataloader(train_dataset, num_workers=num_workers,batch_size=1))
    val_loaders.append(data.build_dataloader(val_dataset, num_workers=num_workers,batch_size=1))
    test_loaders.append(data.build_dataloader(test_dataset, num_workers=num_workers,batch_size=1))


#Create the network
mp = nn.BondMessagePassing(depth=3) # Make the gnn
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
ffn = nn.RegressionFFN() # regression head

# I haven't experimented with this at all, not sure if it will affect the SSL
batch_norm = False

#initialize the model
mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

#fp_mpnn = models.MPNN.load_from_checkpoint('/home/bhanu/Documents/Chemprop_Models/default_fingerprint_model/checkpoints/best.ckpt')
mpnn.to(torch.device("cpu"))


def clone_weights(model):
    #GNN Weights
    weights=[w.clone() for w in model.parameters()]
    
    return weights


# Define the loss function and optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(mpnn.parameters(), lr = 0.001)

def message(H, bmg):
    index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
        0, index_torch, H, reduce="sum", include_self=False
    )[bmg.edge_index[0]]
    M_rev = H[bmg.rev_edge_index]

    return M_all - M_rev

def update(M_t, H_0, weights):
    """Calcualte the updated hidden for each edge"""
    H_t = F.linear(M_t, weights, None)
    H_t = F.relu(H_0 + H_t)

    return H_t

def finalize(M, V, weights, biases):
    H = F.linear(torch.cat((V, M), dim=1), weights, biases)  # V x d_o
    H = F.relu(H)

    return H

# Output without using model
def argforward(weights, bmg):
    H_0 = F.linear(torch.cat([bmg.V[bmg.edge_index[0]],bmg.E],dim=1), weights[0],None)
    H = F.relu(H_0)

    for i in range(3):
        M = message(H,bmg)
        H = update(M,H_0, weights[1])


    index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
    H_v = finalize(M,bmg.V,weights[2], weights[3])
    H = agg(H_v, bmg.batch)

    output = F.linear(H,weights[4],weights[5])
    output = F.relu(output)
    output = F.linear(output,weights[6],weights[7])

    return output


# inner optimization loop
for epoch in range(100):
    total_loss = 0
    optimizer.zero_grad()
    for task in tasks:
        train_loader = train_loaders[task]
        val_loader = val_loaders[task]
        test_loader = test_loaders[task]

        temp_weights = clone_weights(mpnn)
        for batch in train_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            pred=argforward(temp_weights, bmg)

            targets = targets.reshape(-1,1)
            loss = criterion(pred, targets)
            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-0.001*g for w,g in zip(temp_weights,grads)] #temporary update of weights

        for batch in val_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            pred=argforward(temp_weights, bmg)

            targets = targets.reshape(-1,1)
            metaloss = criterion(pred, targets)
            total_loss += metaloss


    metagrads=torch.autograd.grad(total_loss,mpnn.parameters())
    #important step
    for w,g in zip(mpnn.parameters(),metagrads):
        w.grad=g
    
    optimizer.step()
    print(epoch, total_loss)


final_preds = []
final_targets = []

for task in tasks:
    pred_out = []
    target_out = []
    temp_weights=[w.clone() for w in mpnn.parameters()]

    train_loader = train_loaders[task]
    val_loader = val_loaders[task]
    test_loader = test_loaders[task]

    
    for batch in train_loader:
        for i in range(5):
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            pred=argforward(temp_weights, bmg)

            targets = targets.reshape(-1,1)
            loss = criterion(pred, targets)
            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-0.00001*g for w,g in zip(temp_weights,grads)] #temporary update of weights
    
    test_loss = 0
    for batch in test_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        pred=argforward(temp_weights, bmg)

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



for task in tasks:
    pred_label = 'pred_' + str(task)
    true_label = 'true_' + str(task)
    df = pd.DataFrame({true_label:final_targets[task], pred_label:final_preds[task]})

    filename = str(task) + '.csv'
    df.to_csv(filename)

