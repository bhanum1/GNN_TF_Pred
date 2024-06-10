# Import Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats
import math

# Which graphs to make
parity_plot = False
training_curve = True



# Add PATH here
PATH = '/Users/bhanumamillapalli/Desktop/'

# Plot style
plt.style.use("Solarize_Light2")

if parity_plot:
    # Read the csv
    df = pd.read_csv(PATH + "parity.csv")

    truths = df['viscosity_true']
    preds = df['viscosity_pred']

    T1_truths = truths[:55]
    T1_preds = preds[:55]

    T2_truths = truths[55:110]
    T2_preds = preds[55:110]

    T3_truths = truths[110:165]
    T3_preds = preds[110:165]

    T4_truths = truths[165:220]
    T4_preds = preds[165:220]

    T5_truths = truths[220:275]
    T5_preds = preds[220:275]

    truths = [T1_truths, T2_truths, T3_truths, T4_truths, T5_truths]
    preds = [T1_preds, T2_preds, T3_preds, T4_preds, T5_preds]

    for i in range(5):
        #parity plot
        plt.figure()
        plt.plot(truths[i],truths[i],"xkcd:dusty pink")
        plt.plot(truths[i],preds[i], "xkcd:wine", linestyle = '', marker = '.')
        plt.title("Parity Plot for Temp " + str(i+1))

        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.text(
        min(truths[i]),
        max(truths[i]),
        f"Corr = {scipy.stats.spearmanr(truths[i][:], preds[i][:])[0]:.3f}",
        )
        plt.show()

if training_curve:
    # Read the csv
    df = pd.read_csv(PATH + "metrics.csv")
    
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
    plt.show()