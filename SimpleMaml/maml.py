#import packages (can do pip install chemprop)
import pandas as pd
import sklearn.decomposition
import sklearn.preprocessing
import numpy as np
from pathlib import Path
from chemprop import data, featurizers, models, nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import scipy
from model import *
from data import get_loaders
import math

def inner_loop(model, inner_lr, task, steps, m_support, k_query, test=False):
    temp_weights = clone_weights(model) #clone weights

    #get loaders with appropriate number of datapoints
    s_loader, q_loader = get_loaders(task, m_support, k_query)

    # train on support data
    for batch in s_loader:
        x,y = batch
        x = x.to(device)
        # Gradient descent
        for i in range(steps):
            pred=argforward(temp_weights, x).to(device)
            y = y.to(device)

            loss = criterion(pred, y).to(device) #MSE

            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-inner_lr*g for w,g in zip(temp_weights,grads)] #temporary update of weights

    #Calculate metaloss on query data
    metaloss = 0
    for batch in q_loader:
        x,y = batch
        x = x.to(device)

        pred=argforward(temp_weights, x).to(device)

        y = y.to(device)
        metaloss += criterion(pred, y).to(device)

    if test:
        return x,pred,y
    else:
        return metaloss


def outer_loop(model, inner_lr, tasks, m_support, k_query):
    total_loss = 0
    for task in tasks:
        metaloss = inner_loop(model,inner_lr, task, steps=1 ,m_support=m_support, k_query=k_query)
        total_loss+= metaloss
    
    return total_loss / len(tasks)


def train(model, num_epochs, optimizer, num_train, train_tasks, inner_lr, m_support, k_query):
    #training loop
    train_curve = []
    avg_loss = 0
    max_lr = inner_lr
    for epoch in range(num_epochs):
        inner_lr = max_lr * pow(0.95,epoch/20000)
        optimizer.zero_grad()
        #sample collection of tasks to train on
        sample_amplitudes, sample_phases = random.sample(train_tasks[0], num_train), random.sample(train_tasks[1], num_train)

        task_sample = []
        for i in range(len(sample_amplitudes)):
            pair = (sample_amplitudes[i], sample_phases[i])
            task_sample.append(pair)

        #run loops and get metaloss
        metaloss = outer_loop(model, inner_lr, task_sample, m_support, k_query)

        #backpropagate
        metagrads=torch.autograd.grad(metaloss,model.parameters())
        #important step
        for w,g in zip(model.parameters(),metagrads):
            w.grad=g
        
        optimizer.step()
        scheduler.step()

        avg_loss += metaloss.cpu().detach().numpy()

        if epoch == 0 or (epoch+1) % 1000 == 0:
            print("{0} Avg Train Loss (last 100): {1:.3f}".format(epoch, avg_loss/1000))
            train_curve.append(avg_loss/1000)
            avg_loss = 0
            
            

    dict = {"Train_loss":train_curve}
    curve = pd.DataFrame(dict)
    curve.to_csv('training_curve.csv')


def eval(model, fine_lr, fine_tune_steps, test_task, m_support, k_query):

    pred_out = []
    target_out = []
    x, pred, target = inner_loop(model, fine_lr, test_task, fine_tune_steps, m_support, k_query, True)
    loss = criterion(pred, target)

    x,pred,target = x.squeeze().cpu().detach().numpy(),pred.squeeze().cpu().detach().numpy(), target.squeeze().cpu().detach().numpy()

    print("Amplitude: {0:.3f} Phase: {1:.3f} MSE:{2:.3f}".format(test_task[0], test_task[1],loss))

    dict = {"X":x, "True":target, "Pred":pred}
    df = pd.DataFrame(dict)
    df.to_csv('pred.csv')
    

# Define the loss function and optimizer
meta_lr = 0.01
inner_lr = 0.01
fine_lr = 0.01
fine_tune_steps = 10
epochs = 1000000
m_support = 10
k_query = 10
num_train_sample = 10

criterion = torch.nn.L1Loss(reduction='mean')

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize the model
ffn = SinusoidModel(1)
ffn.to(device)
optimizer = optim.Adam(ffn.parameters(), lr = meta_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10000,0.95)

#create list of train tasks
phases = list(np.linspace(0, math.pi, 100))
amplitudes = list(np.linspace(0.1,5, 100))

train_tasks = [amplitudes, phases]

#eval(mpnn, test_data, fine_lr, fine_tune_steps, combo, m_support=10, k_query=1)
train(ffn, epochs, optimizer, num_train_sample, train_tasks, inner_lr,m_support,k_query)
eval(ffn, fine_lr, fine_tune_steps,(1,0), m_support=10, k_query=100)


'''
directory = 'results'


results = []
result_dict = dict()
for file in os.scandir(directory):
    nums = file.path[-7:-4]

    if nums.isnumeric():
        df = pd.read_csv(file)

        label1 = nums[0] + "_" + nums
        label2 = nums[1] + "_" + nums
        #label3 = nums[2] + "_" + nums

        #rcc1 = scipy.stats.spearmanr(df['true_' + nums[0]], df['pred_' + nums[0]])
        #srcc2 = scipy.stats.spearmanr(df['true_' + nums[1]], df['pred_' + nums[1]])

        MAE1 = np.average(abs(df['true_' + nums[0]]-df['pred_'+nums[0]]))
        MAE2 = np.average(abs(df['true_' + nums[1]]-df['pred_'+nums[1]]))
        MAE3 = np.average(abs(df['true_' + nums[2]]-df['pred_'+nums[2]]))


        #result_dict[label1] = srcc1[0]
        #result_dict[label2] = srcc2[0]

        result_dict[label1] = MAE1
        result_dict[label2] = MAE2
        result_dict[label3] = MAE3

df = pd.DataFrame.from_dict(result_dict, orient='index', columns = ['MAE'])

filename = 'test.csv'
df.to_csv(filename)
'''
