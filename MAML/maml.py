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
from itertools import combinations
import os
import scipy
from model import *
from data import generate_data, get_loaders


datafile = "data/train_data.csv"
test_datafile = 'data/test_data.csv'
checkpoint_path = 'checkpoints/best.ckpt'

def inner_loop(model, inner_lr, task_data, task, steps, m_support, k_query, test_indices= None):
    temp_weights = clone_weights(model) #clone weights

    #get loaders with appropriate number of datapoints
    s_loader, q_loader, test_loader = get_loaders(task_data[task], m_support, k_query, test_indices)
    # train on support data
    for batch in s_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg.to(device)
        
        # Gradient descent
        for i in range(steps):
            pred=argforward(temp_weights, bmg).to(device)
            targets = targets.reshape(-1,1).to(device)

            loss = criterion(pred, targets).to(device) #MSE

            grads=torch.autograd.grad(loss,temp_weights)
            temp_weights=[w-inner_lr*g for w,g in zip(temp_weights,grads)] #temporary update of weights

            
            if test_indices is not None:
                for t_batch in test_loader:
                    t_bmg, V_d, X_d, t_targets, weights, lt_mask, gt_mask = t_batch
                    t_bmg.to(device)

                    t_pred=argforward(temp_weights, t_bmg).to(device)

                    t_targets = t_targets.reshape(-1,1).to(device)

                print("supp MAE: {0:.3f} Test MAE: {1:.3f}".format(np.average(abs(pred.cpu().detach().numpy() - targets.cpu().detach().numpy())), np.average(abs(t_pred.cpu().detach().numpy() - t_targets.cpu().detach().numpy()))))



    if test_indices is None:
        #Calculate metaloss on query data
        metaloss = 0
        for batch in q_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            bmg.to(device)

            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)
            metaloss += criterion(pred, targets).to(device)


        # validation loss
        task_val = val(temp_weights, val_data, fine_lr, task)

        return metaloss, task_val
    else:
        for batch in test_loader:
            bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
            bmg.to(device)

            pred=argforward(temp_weights, bmg).to(device)

            targets = targets.reshape(-1,1).to(device)

        return pred, targets

def outer_loop(model, inner_lr, task_data, tasks, m_support, k_query):
    total_loss = 0
    total_val_loss = 0
    for task in tasks:
        metaloss, task_val = inner_loop(model,inner_lr, task_data, task, steps=1 ,m_support=m_support, k_query=k_query)
        total_loss+= metaloss
        total_val_loss += task_val
    
    return total_loss / len(tasks), total_val_loss / len(tasks)


def train(model, num_epochs, optimizer, num_train, task_data, train_tasks, inner_lr, m_support, k_query):
    #training loop
    train_curve = []
    val_curve = []

    lr_max = inner_lr
    total_loss, total_val_loss = 0,0
    for epoch in range(num_epochs):
        inner_lr = lr_max * pow(2,-epoch/1000)

        optimizer.zero_grad()
        #sample collection of tasks to train on
        task_sample = random.sample(train_tasks, num_train)

        #run loops and get metaloss
        metaloss, val_loss = outer_loop(model, inner_lr, task_data, task_sample, m_support, k_query)

        #backpropagate
        metagrads=torch.autograd.grad(metaloss,model.parameters())
        #important step
        for w,g in zip(model.parameters(),metagrads):
            w.grad=g
        
        optimizer.step()
        scheduler1.step()

        total_loss += metaloss.cpu().detach().numpy()
        total_val_loss += val_loss.cpu().detach().numpy()

        if epoch == 0 or (epoch+1) % 100 == 0:
            print("{0} Train Loss: {1:.3f} Val Loss: {2:.3f}".format(epoch, total_loss / 100, total_val_loss / 100))
            train_curve.append(total_loss/100)
            val_curve.append(total_val_loss/100)

            total_loss, total_val_loss = 0,0

    dict = {"Train_loss":train_curve, "Val_loss":val_curve}
    curve = pd.DataFrame(dict)
    curve.to_csv('training_curve.csv')



def val(model_weights, task_data, fine_lr, task):
    s_loader, _, _ = get_loaders(task_data[task], 50, 1, None)

    for batch in s_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        bmg.to(device)
        pred=argforward(model_weights, bmg).to(device)
        targets = targets.reshape(-1,1).to(device)

        loss = criterion(pred, targets).to(device) #MSE
    
    return loss



def eval(model, task_data, fine_lr, fine_tune_steps, test_tasks, m_support, k_query):
    final_preds = []
    final_targets = []
    for task in test_tasks:

        test_indices = random.sample(range(len(task_data[task])), 50)
        pred_out = []
        target_out = []

        pred, target = inner_loop(model, fine_lr, task_data, task, fine_tune_steps, m_support, k_query, test_indices)
        

        pred,target = pred.cpu().detach().numpy(), target.cpu().detach().numpy()
        pred_out.extend(pred)
        target_out.extend(target)

        
        print("Task:{0} MAE:{1:.3f}, R^2:{2:.3f}".format(task, np.average(abs(np.array(pred_out) - np.array(target_out))), round(scipy.stats.spearmanr(pred, target)[0],3)))

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

    filename = 'results/' + str(test_tasks[0]) + "_" + str(test_tasks[1]) + '.csv'
    #filename = 'results/' + str(test_tasks[0]) + "_" + str(test_tasks[1]) + "_" + str(test_tasks[2]) + '.csv'
    df.to_csv(filename)
    

# Define the loss function and optimizer
meta_lr = 0.01
inner_lr = 0.001
fine_lr = 0.01
fine_tune_steps = 100
epochs = 10000
m_support = 10
k_query = 10
num_train_sample = 3

criterion = torch.nn.MSELoss(reduction='mean')

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train all combinations
comb = list(combinations(range(9),2))
combos = random.sample(comb, 1)


for combo in combos:
    #initialize the model
    mpnn = build_model()
    mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
    mpnn.to(device)
    optimizer = optim.Adam(mpnn.parameters(), lr = meta_lr)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer,1000,0.5)
    
    #create list of train tasks
    train_tasks = []
    for i in range(9):
        if i not in combo:
            train_tasks.append(i)

    task_data, val_data = generate_data(datafile,train_tasks)
    test_data, _ = generate_data(datafile, combo, test=True)

    #eval(mpnn, test_data, fine_lr, fine_tune_steps, combo, m_support=10, k_query=1)
    train(mpnn, epochs, optimizer, num_train_sample,task_data, train_tasks, inner_lr,m_support,k_query)
    eval(mpnn, test_data, fine_lr, fine_tune_steps, combo, m_support=10, k_query=1)

    torch.save(mpnn.state_dict(), 'results/model.pt')


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
