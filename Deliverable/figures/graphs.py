# Import Libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats
import math

# Which graphs to make
parity_plot = False
training_curve = False
summary = True
quantity = "vapor_pressure"

columns = ['T1', 'T2', 'T3', 'T4', 'T5']
# Add data PATH here
PATH = '/home/bhanu/Documents/GitHub/Thermal_Fluid_Prediction_GNN/Deliverable/figures/data/'

# Plot style
plt.style.use("Solarize_Light2")

if parity_plot:
    # Read the csv
    df = pd.read_csv(PATH + quantity + "_parity.csv")
    
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
        MAE = np.abs(np.subtract(truths[i],preds[i])).mean()
        plt.text(
        min(truths[i]),
        max(preds[i]),
        f"SRCC = {scipy.stats.spearmanr(truths[i][:], preds[i][:])[0]:.3f}, MSE = {MSE:.3f}, MAE = {MAE:.3f}",
        )
        plt.savefig(quantity + "/parity_" + str(columns[i]) + ".png")
        plt.show()

if training_curve:
    # Read the csv
    df = pd.read_csv(PATH + quantity + "_training.csv")
    
    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']

    #remove zeros
    train_epochs = list(epochs)
    val_epochs = list(epochs)
    train_deletions = []
    val_deletions = []
    
    #outlier removal
    mean_train = np.mean(train_loss)
    std_train = np.std(train_loss)
    mean_val = np.mean(val_loss)
    std_val = np.std(val_loss)

    for i in range(len(train_loss)):
        if math.isnan(train_loss[i]) or train_loss[i] > mean_train + 3 * std_train:
            train_deletions.append(i)
        if math.isnan(val_loss[i]) or val_loss[i] > mean_val + 3 * std_val:
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
    plt.savefig(quantity + "/training_curve.png")
    plt.show()
   
if summary:
    df = pd.read_csv(PATH + "summary.csv")
    barWidth = 0.2
    fig, ax = plt.subplots(figsize = (8,5))
    ax2 = ax.twinx()

    srcc = df['SRCC']
    mae = df['MAE / scale']

    br1 = np.arange(len(srcc))
    br2 = [x + barWidth for x in br1]

    ax.bar(br1, srcc, width = barWidth, color = 'xkcd:dusty pink', label = 'SRCC')
    ax2.bar(br2, mae, width = barWidth, color = 'xkcd:wine', label = 'MAE / Scale')
    ax2.yaxis.grid(False)


    plt.xlabel('Metric')
    ax.set_ylabel('SRCC')
    ax2.set_ylabel('MAE / Scale')
    plt.xticks([r + barWidth for r in range(len(srcc))], ['Dynamic Viscosity', 'Thermal Conductivity', 'Vapor Pressure', 'Density'])
    plt.title("Comparison of Model Performance for Properties")
    
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h = h1 + h2
    l = l1 + l2

    ax.legend(h, l)


    plt.show()