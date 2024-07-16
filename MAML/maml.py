#import packages (can do pip install chemprop)
import pandas as pd
import numpy as np
from pathlib import Path
from chemprop import data, featurizers, models, nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from itertools import combinations
import os
import scipy

'''
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.

def generate_data(csv, test):
    input_path = csv # path to your data .csv file
    df = pd.read_csv(input_path) #convert to dataframe
    num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading

    #get columns
    smis = df['smiles']
    targets = df['target']
    task_labels = df['task']

    #create holders for all dataloaders
    train_s_loaders = []
    train_q_loaders = []
    test_s_loaders = []
    test_q_loaders = []
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    for task in range(10):
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

        #add to loader list
        if task in test:
            test_s_loaders.append(data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=10))
            test_q_loaders.append(data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=10))
        else:
            train_s_loaders.append(data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=10))
            train_q_loaders.append(data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=10))
        
    return [(train_s_loaders, train_q_loaders), (test_s_loaders, test_q_loaders)]


def clone_weights(model):
    #GNN Weights
    weights=[w.clone() for w in model.parameters()]
    
    return weights

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

def inner_loop(model, inner_lr, train_loaders, grad_steps, eval= False):
    s_loader, q_loader = train_loaders
    
    temp_weights = clone_weights(model)
        
    for batch in s_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        
        bmg.to(device)
        for i in range(grad_steps):
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

    if not eval:
        return metaloss
    else:
        return metaloss, pred, targets

def outer_loop(model, num_epochs, optimizer, train_loaders, inner_lr):
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for task in range(len(train_loaders[0])):
            s_loader, q_loader = train_loaders[0][task], train_loaders[1][task]

            metaloss = inner_loop(model,inner_lr, [s_loader, q_loader],1)
            total_loss+= metaloss
        
        metagrads=torch.autograd.grad(total_loss,model.parameters())
        #important step
        for w,g in zip(model.parameters(),metagrads):
            w.grad=g
        
        optimizer.step()
        if epoch == 0 or (epoch+1) % 100 == 0:
            print("{0} Train Loss: {1:.3f}".format(epoch, total_loss.cpu().detach().numpy() / 8))

def eval(model, fine_lr, test_loaders, test_tasks):
    final_preds = []
    final_targets = []

    total_loss = 0

    for task in range(len(test_loaders[0])):
        pred_out = []
        target_out = []
        s_loader, q_loader = test_loaders[0][task], test_loaders[1][task]

        test_loss, pred, target = inner_loop(model, fine_lr, [s_loader, q_loader], 20, True)
        pred_out.extend(pred.cpu().detach().numpy())
        target_out.extend(target.cpu().detach().numpy())
        
        total_loss += test_loss


        print(task, total_loss)

        for i in range(len(pred_out)):
            pred_out[i] = pred_out[i][0]
        for i in range(len(target_out)):
            target_out[i] = target_out[i][0]

        final_preds.append(pred_out)
        final_targets.append(target_out)

    out_dict = dict()
    for task in range(len(test_tasks)):
        pred_label = 'pred_' + str(test_tasks[task])
        true_label = 'true_' + str(test_tasks[task])

        out_dict[true_label] = final_targets[task]
        out_dict[pred_label] = final_preds[task]

    df = pd.DataFrame(out_dict)

    filename = 'results/' + str(test_tasks[0]) + str(test_tasks[1]) + '.csv'
    df.to_csv(filename)


loaders = generate_data("maml_bigdata.csv", [0,1])



def build_model():
        
    #Create the network
    mp = nn.BondMessagePassing(depth=3) # Make the gnn
    agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
    ffn = nn.RegressionFFN() # regression head

    # I haven't experimented with this at all, not sure if it will affect the SSL
    batch_norm = False

    #initialize the model
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

    return mpnn


# Define the loss function and optimizer
outer_lr = 0.00005
inner_lr = 0.0005
fine_lr = 0.001
epochs = 500

criterion = torch.nn.MSELoss(reduction='mean')

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


comb = combinations(range(10),2)

for combo in list(comb):
    #initialize the model
    mpnn = build_model()
    mpnn.to(device)
    optimizer = optim.Adam(mpnn.parameters(), lr = outer_lr)

    loaders = generate_data("maml_bigdata.csv", combo)
    outer_loop(mpnn, epochs, optimizer, loaders[0], inner_lr)
    eval(mpnn, fine_lr, loaders[1], combo)
'''

directory = 'results'


results = []
result_dict = dict()
for file in os.scandir(directory):
    nums = file.path[-6:-4]

    if nums.isnumeric():
        df = pd.read_csv(file)

        label1 = nums[0] + "_" + nums
        label2 = nums[1] + "_" + nums

        #rcc1 = scipy.stats.spearmanr(df['true_' + nums[0]], df['pred_' + nums[0]])
        #srcc2 = scipy.stats.spearmanr(df['true_' + nums[1]], df['pred_' + nums[1]])

        MAE1 = np.average(abs(df['true_' + nums[0]]-df['pred_'+nums[0]]))
        MAE2 = np.average(abs(df['true_' + nums[1]]-df['pred_'+nums[1]]))


        #result_dict[label1] = srcc1[0]
        #result_dict[label2] = srcc2[0]

        result_dict[label1] = MAE1
        result_dict[label2] = MAE2

df = pd.DataFrame.from_dict(result_dict, orient='index', columns = ['MAE'])

filename = 'test.csv'
df.to_csv(filename)

