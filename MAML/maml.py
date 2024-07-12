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
import random


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
nontrain = random.sample(range(num_tasks),4)
test_tasks = nontrain[:2]
val_tasks = nontrain[2:]
tasks = []

for task in range(num_tasks):
    if task not in nontrain:
        tasks.append(task)

#create holders for all dataloaders
train_s_loaders = []
train_q_loaders = []
val_s_loaders = []
val_q_loaders = []
test_s_loaders = []
test_q_loaders = []
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


    indices=list(range(50))
    random.shuffle(indices)
    supp_indices = indices[:40]
    query_indices = indices[40:]

    supp_data, query_data, _ = data.split_data_by_indices(task_data, supp_indices, query_indices,None)
    
    supp_dataset = data.MoleculeDataset(supp_data, featurizer)
    query_dataset = data.MoleculeDataset(query_data, featurizer)

    train_s_loaders.append(data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=10))
    train_q_loaders.append(data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=10))

for task in val_tasks:
    indices = task_labels == task
    task_smis = smis.loc[indices]
    task_targets = targets.loc[indices]

    #create data
    task_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis, task_targets)]
    mols = [d.mol for d in task_data]

    indices=list(range(50))
    random.shuffle(indices)
    supp_indices = indices[:40]
    query_indices = indices[40:]

    supp_data, query_data, _ = data.split_data_by_indices(task_data, supp_indices, query_indices,None)
    
    supp_dataset = data.MoleculeDataset(supp_data, featurizer)
    query_dataset = data.MoleculeDataset(query_data, featurizer)

    val_s_loaders.append(data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=10))
    val_q_loaders.append(data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=10))

for task in test_tasks:
    indices = task_labels == task
    task_smis = smis.loc[indices]
    task_targets = targets.loc[indices]

    #create data
    task_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis, task_targets)]
    mols = [d.mol for d in task_data]

    indices=list(range(50))
    random.shuffle(indices)
    supp_indices = indices[:40]
    query_indices = indices[40:]

    supp_data, query_data, _ = data.split_data_by_indices(task_data, supp_indices, query_indices,None)
    
    supp_dataset = data.MoleculeDataset(supp_data, featurizer)
    query_dataset = data.MoleculeDataset(query_data, featurizer)

    test_s_loaders.append(data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=10))
    test_q_loaders.append(data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=10))


#Create the network
mp = nn.BondMessagePassing(depth=3) # Make the gnn
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
ffn = nn.RegressionFFN() # regression head

# I haven't experimented with this at all, not sure if it will affect the SSL
batch_norm = False

#initialize the model
mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

#fp_mpnn = models.MPNN.load_from_checkpoint('/home/bhanu/Documents/Chemprop_Models/default_fingerprint_model/checkpoints/best.ckpt')


def clone_weights(model):
    #GNN Weights
    weights=[w.clone() for w in model.parameters()]
    
    return weights


# Define the loss function and optimizer
def lr(epoch):
    scalar = epoch//100
    outer_lr = 0.0005 / scalar
    inner_lr = 0.001 / scalar

    return outer_lr, inner_lr

outer_lr = 0.01
inner_lr = 0.01
fine_lr = 0.005
epochs = 300
finetune_steps = 5

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = optim.SGD(mpnn.parameters(), lr = outer_lr)

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

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mpnn.to(device)


# inner optimization loop
for epoch in range(epochs):
    total_loss = 0
    val_loss = 0
    optimizer.zero_grad()
    for task in range(len(train_s_loaders)):
        s_loader = train_s_loaders[task]
        q_loader = train_q_loaders[task]

        val_loader = val_loaders[task]

        temp_weights = clone_weights(mpnn)
        
        for batch in s_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            
            bmg.to(device)
            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)
            loss = criterion(pred, targets).to(device)
            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-inner_lr*g for w,g in zip(temp_weights,grads)] #temporary update of weights

        for batch in q_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            bmg.to(device)
            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)
            metaloss = criterion(pred, targets).to(device)
            total_loss += metaloss
        
        for batch in val_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            bmg.to(device)
            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)
            val_metaloss = criterion(pred, targets).to(device)
            val_loss += val_metaloss


    metagrads=torch.autograd.grad(total_loss,mpnn.parameters())
    #important step
    for w,g in zip(mpnn.parameters(),metagrads):
        w.grad=g
    
    optimizer.step()
    print("{0} Train Loss: {1:.3f} Val Loss: {2:.3f}".format(epoch, total_loss.detach().numpy() / 8, val_loss.detach().numpy() / 10))


final_preds = []
final_targets = []

for task in range(len(test_s_loaders)):
    pred_out = []
    target_out = []
    temp_weights=[w.clone() for w in mpnn.parameters()]

    s_loader = train_s_loaders[task]
    q_loader = train_q_loaders[task]

    
    for batch in s_loader:
        for i in range(finetune_steps):
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            bmg.to(device)
            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)
            loss = criterion(pred, targets)
            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-fine_lr*g for w,g in zip(temp_weights,grads)] #temporary update of weights
    
    test_loss = 0
    for batch in q_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg.to(device)
        pred=argforward(temp_weights, bmg)

        targets = targets.reshape(-1,1).to(device)
        test_loss += criterion(pred, targets)
        pred = pred.cpu()
        targets = targets.cpu()
        pred_out.extend(pred.detach().numpy())
        target_out.extend(targets.detach().numpy())

    print(task, test_loss)
    for i in range(len(pred_out)):
        pred_out[i] = pred_out[i][0]
    for i in range(len(target_out)):
        target_out[i] = target_out[i][0]

    final_preds.append(pred_out)
    final_targets.append(target_out)



for task in range(len(test_tasks)):
    pred_label = 'pred_' + str(task)
    true_label = 'true_' + str(task)
    df = pd.DataFrame({true_label:final_targets[task], pred_label:final_preds[task]})

    filename = str(test_tasks[task]) + '.csv'
    df.to_csv(filename)