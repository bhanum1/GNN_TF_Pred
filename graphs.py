# Import Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats
import math

# Which graphs to make
parity_plot = True
training_curve = True
quantity = "viscosity"

columns = ['T1', 'T2', 'T3', 'T4', 'T5']
# Add PATH here
PATH = '/home/bhanu/Documents/GitHub/Thermal_Fluid_Prediction_GNN/Graph_Data/'

# Plot style
plt.style.use("Solarize_Light2")

if parity_plot:
    # Read the csv
    df = pd.read_csv(PATH + quantity + "/parity.csv")
    
    truths, preds = [], []
    for column in columns:
        true_col = df[column + '_true'].dropna()
        pred_col = df[column + '_pred'].dropna()

        truths.append(true_col)
        preds.append(pred_col)

    for i in range(len(truths)):
        #parity plot
        plt.figure()
        plt.plot(truths[i],truths[i],"xkcd:dusty pink")
        plt.plot(truths[i],preds[i], "xkcd:wine", linestyle = '', marker = '.')
        plt.title("Parity Plot for " + str(columns[i]))

        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        MSE = np.square(np.subtract(truths[i],preds[i])).mean() 
        plt.text(
        min(truths[i]),
        max(preds[i]),
        f"SRCC = {scipy.stats.spearmanr(truths[i][:], preds[i][:])[0]:.3f}, MSE = {MSE:.3f}",
        )
        plt.savefig("Graph_Data/" + quantity + "/parity_" + str(columns[i]) + ".png")
        plt.show()

if training_curve:
    # Read the csv
    df = pd.read_csv(PATH + quantity + "/training.csv")
    
    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    #remove zeros
    train_epochs = list(epochs)
    val_epochs = list(epochs)
    train_deletions = []
    val_deletions = []
    for i in range(len(train_loss)):
        if math.isnan(train_loss[i]):
            train_deletions.append(i)
        if math.isnan(val_loss[i]):
            val_deletions.append(i)

    for index in sorted(train_deletions, reverse=True):
        del train_loss[index]
        del train_epochs[index]

    for index in sorted(val_deletions, reverse=True):
        del val_loss[index]
        del val_epochs[index]

    #training curve
    plt.figure()
    plt.plot(train_epochs,train_loss, 'xkcd:dusty pink', label = 'Training', linestyle = '-')
    plt.plot(val_epochs, val_loss, 'xkcd:wine', label = 'Validation')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig("Graph_Data/" + quantity + "/training_curve.png")
    plt.show()
   
