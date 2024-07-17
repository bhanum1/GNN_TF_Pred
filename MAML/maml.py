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
from model import *
from data import generate_data

datafile = "maml_final.csv"

def inner_loop(model, inner_lr, task, grad_steps, m_support, k_query, eval= False):
    temp_weights = clone_weights(model) #clone weights

    #get loaders with appropriate number of datapoints
    s_loader, q_loader = generate_data(datafile,task, m_support, k_query)
        
    # train on support data
    for batch in s_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg.to(device)

        # Gradient descent a defined number of times
        for i in range(grad_steps):
            pred=argforward(temp_weights, bmg).to(device)
            targets = targets.reshape(-1,1).to(device)

            loss = criterion(pred, targets).to(device) #MSE

            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-inner_lr*g for w,g in zip(temp_weights,grads)] #temporary update of weights

    #Calculate metaloss on query data
    metaloss = 0
    for batch in q_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg.to(device)
        pred=argforward(temp_weights, bmg).to(device)

        targets = targets.reshape(-1,1).to(device)
        metaloss += criterion(pred, targets).to(device)

    if not eval:
        return metaloss
    else:
        return metaloss, pred, targets

def outer_loop(model, inner_lr, tasks, m_support, k_query):
    total_loss = 0
    for task in tasks:
        metaloss = inner_loop(model,inner_lr, task, grad_steps=1 ,m_support=m_support, k_query=k_query)
        total_loss+= metaloss
    
    return total_loss

def train(model, num_epochs, optimizer, num_train, train_tasks, inner_lr, m_support, k_query):
    #training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        #sample collection of tasks to train on
        task_sample = random.sample(train_tasks, num_train)

        #run loops and get metaloss
        metaloss = outer_loop(model, inner_lr, task_sample, m_support, k_query)

        #backpropagate
        metagrads=torch.autograd.grad(metaloss,model.parameters())
        #important step
        for w,g in zip(model.parameters(),metagrads):
            w.grad=g
        
        optimizer.step()

        if epoch == 0 or (epoch+1) % 100 == 0:
            print("{0} Train Loss: {1:.3f}".format(epoch, metaloss.cpu().detach().numpy() / len(train_tasks)))

def eval(model, fine_lr, fine_tune_steps, test_tasks, m_support, k_query):
    final_preds = []
    final_targets = []

    for task in test_tasks:
        pred_out = []
        target_out = []

        test_loss, pred, target = inner_loop(model, fine_lr, task, fine_tune_steps, m_support, k_query, True)

        pred,target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()
        pred_out.extend(pred)
        target_out.extend(target)

        print(task, test_loss.cpu().detach().numpy(), round(scipy.stats.spearmanr(pred, target)[0],3))

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

    filename = 'results/' + str(test_tasks[0]) + "_" + str(test_tasks[1]) + "_" + str(test_tasks[2]) + '.csv'
    df.to_csv(filename)


# Define the loss function and optimizer
meta_lr = 0.0001
inner_lr = 0.0001
fine_lr = 0.00005
fine_tune_steps = 2
epochs = 20
m_support = 5
k_query = 20
num_train_sample = 3

criterion = torch.nn.MSELoss(reduction='mean')

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train all combinations
comb = list(combinations(range(15),3))
combos = random.sample(comb, 10)



for combo in combos:
    print(combo)
    #initialize the model
    mpnn = build_model()
    mpnn.to(device)
    optimizer = optim.Adam(mpnn.parameters(), lr = meta_lr)
    
    #create list of train tasks
    train_tasks = []
    for i in range(15):
        if i not in combo:
            train_tasks.append(i)


    train(mpnn, epochs, optimizer, num_train_sample,train_tasks, inner_lr,m_support,k_query)
    eval(mpnn, fine_lr, fine_tune_steps, combo, m_support, k_query)


directory = 'results'


results = []
result_dict = dict()
for file in os.scandir(directory):
    nums = file.path[-6:-4]

    if nums.isnumeric():
        df = pd.read_csv(file)

        label1 = nums[0] + "_" + nums
        label2 = nums[1] + "_" + nums
        label3 = nums[2] + "_" + nums

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

