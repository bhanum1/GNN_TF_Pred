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
        x_q,y_q = batch
        x_q = x_q.to(device)

        pred=argforward(temp_weights, x_q).to(device)

        y_q = y_q.to(device)
        metaloss += criterion(pred, y_q).to(device)

    if test:
        return x_q,pred,y_q, x
    else:
        return metaloss


def outer_loop(model, inner_lr, tasks, m_support, k_query):
    total_loss = 0
    for task in tasks:
        metaloss = inner_loop(model,inner_lr, task, steps=1 ,m_support=m_support, k_query=k_query)
        total_loss+= metaloss
    
    return total_loss / len(tasks)


def train(model, num_epochs, optimizer, num_train, inner_lr, m_support, k_query):
    #training loop
    train_curve = []
    avg_loss = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        #sample collection of tasks to train on

        task_sample = []
        for _ in range(num_train):
            amp = np.random.uniform(0.1, 5)
            phase = np.random.uniform(0, math.pi)
            task_sample.append((amp, phase))

        #run loops and get metaloss
        metaloss = outer_loop(model, inner_lr, task_sample, m_support, k_query)

        #backpropagate
        metagrads=torch.autograd.grad(metaloss,model.parameters())
        #important step
        for w,g in zip(model.parameters(),metagrads):
            w.grad=g
        
        optimizer.step()

        avg_loss += metaloss.cpu().detach().numpy()

        if epoch == 0 or (epoch+1) % 10000 == 0:
            print("{0} Avg Train Loss (last 10000): {1:.3f}".format(epoch, avg_loss/10000))
            train_curve.append(avg_loss/10000)
            avg_loss = 0
            
            

    dict = {"Train_loss":train_curve}
    curve = pd.DataFrame(dict)
    curve.to_csv('training_curve.csv')


def eval(model, fine_lr, fine_tune_steps, test_task, m_support, k_query):
    pred_out = []
    target_out = []
    x, pred, target, supp_x = inner_loop(model, fine_lr, test_task, fine_tune_steps, m_support, k_query, True)
    loss = criterion(pred, target)

    x,pred,target, supp_x = (x.squeeze().cpu().detach().numpy(),
                            pred.squeeze().cpu().detach().numpy(), 
                            target.squeeze().cpu().detach().numpy(),
                            supp_x.squeeze().cpu().detach().numpy())

    print("Amplitude: {0:.3f} Phase: {1:.3f} MSE:{2:.3f}".format(test_task[0], test_task[1],loss))

    dict = {"X":x, "True":target, "Pred":pred}
    dict2 = {"Supp_X":supp_x, "filler":np.zeros(len(supp_x))}
    df = pd.DataFrame(dict)
    df.to_csv('pred.csv')
    
    df2 = pd.DataFrame(dict2)
    df2.to_csv('points.csv')

# Define the loss function and optimizer
meta_lr = 0.001
inner_lr = 0.01
fine_lr = 0.01
fine_tune_steps = 10
epochs = 5000000
m_support = 5
k_query = 5
num_train_sample = 10

criterion = torch.nn.L1Loss(reduction='mean')

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize the model
ffn = SinusoidModel(1)
ffn.to(device)
optimizer = optim.Adam(ffn.parameters(), lr = meta_lr)

#create list of train tasks

train(ffn, epochs, optimizer, num_train_sample, inner_lr,m_support,k_query)
eval(ffn, fine_lr, fine_tune_steps,(1,0), m_support=10, k_query=100)
