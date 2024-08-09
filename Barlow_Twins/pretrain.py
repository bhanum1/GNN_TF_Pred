from chemprop import data, featurizers, models, nn
import torch
import torch.nn.functional as F
from data_processing import *
import torch.optim as optim
from BT_loss import BT_loss
from rdkit import RDLogger
from plots import plot_losses
from torch.optim.lr_scheduler import CosineAnnealingLR

RDLogger.DisableLog('rdApp.*')
batch_size = 64

#Create the network
dropout = 0.15
mp_1 = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 700, activation = 'selu')
mp_2 = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 700, activation = 'selu') # Make the gnn
ffn = nn.RegressionFFN(dropout=dropout, input_dim = 700, hidden_dim = 1000, n_layers=1) # regression head
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
batch_norm = False

#initialize the model
mpnn_1 = models.MPNN(mp_1, agg, ffn, batch_norm, [nn.metrics.MSEMetric()])
mpnn_2 = models.MPNN(mp_2, agg, ffn, batch_norm, [nn.metrics.MSEMetric()])

opt1 = optim.SGD(mpnn_1.parameters(), lr = 0.0005, weight_decay=eval('5E-5'))
opt2 = optim.SGD(mpnn_2.parameters(), lr = 0.0005, weight_decay=eval('5E-5'))


input_path = 'Barlow_Twins/dataset.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

smis = df['smiles']


mpnn_1.to(device)
mpnn_2.to(device)

epochs = 200

warmup = 50

sch1 = CosineAnnealingLR(
            opt1, T_max=epochs-warmup, 
            eta_min=0, last_epoch=-1
        )
sch2 = CosineAnnealingLR(
            opt2, T_max=epochs-warmup, 
            eta_min=0, last_epoch=-1
        )

best_val_loss = float('inf')
on_diag_losses, off_diag_losses, losses, val_losses = [], [], [], []
for epoch in range(epochs):
    train, val, train_2, val_2 = create_loaders(smis, batch_size=batch_size, num_workers=0)

    train_l = 0
    on_diag_sum = 0
    off_diag_sum = 0
    for batch_1, batch_2 in zip(train, train_2):
        bmg_1, *_ = batch_1
        bmg_2, *_ = batch_2
        
        bmg_1.to(device)
        bmg_2.to(device)

        r_1 = mpnn_1.message_passing.forward(bmg_1)
        r_1 = agg.forward(r_1, bmg_1.batch)
        r_2 = mpnn_2.message_passing.forward(bmg_2)
        r_2 = agg.forward(r_2, bmg_2.batch)

        

        on_diag, off_diag, loss = BT_loss(r_1, r_2, [1, 0.005])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(mpnn_1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(mpnn_2.parameters(), 1)


        opt1.step()
        opt2.step()

        train_l += loss
        on_diag_sum+= on_diag
        off_diag_sum += off_diag

    print(torch.max(r_1))


    val_loss = 0
    val_on, val_off = 0, 0
    for batch_1, batch_2 in zip(val, val_2):
        bmg_1, V_d_1, X_d, targets, weights, lt_mask, gt_mask, _, _, _ = batch_1
        bmg_2, V_d_2, X_d, targets, weights, lt_mask, gt_mask, _, _, _ = batch_2

        bmg_1.to(device)
        bmg_2.to(device)
        
        r_1 = mpnn_1.message_passing.forward(bmg_1)
        r_1 = agg.forward(r_1, bmg_1.batch)
        r_2 = mpnn_2.message_passing.forward(bmg_2)
        r_2 = agg.forward(r_2, bmg_2.batch)

        on_diag, off_diag, loss = BT_loss(r_1, r_2, [1, 0.01])

        val_loss += loss
        val_on += on_diag.cpu().detach().numpy()
        val_off += off_diag.cpu().detach().numpy()
    

    val_size = int(np.floor(len(smis) * 0.1) / batch_size)

    on_diag_losses.append(on_diag_sum.cpu().detach().numpy() / (val_size * 9))
    off_diag_losses.append(off_diag_sum.cpu().detach().numpy() / (val_size * 9))
    losses.append(train_l.cpu().detach().numpy() / (val_size * 9))
    val_losses.append(val_loss.cpu().detach().numpy() / val_size)

    print(epoch, "Train Loss: {0:.2f} Val Loss: {1:.2f}".format(train_l / (val_size * 9), val_loss / val_size))
    print("Train On:", round(on_diag_losses[epoch],3), "Off:", round(off_diag_losses[epoch],3))
    print("Val On:", round(val_on / val_size,3), "Off:", round(val_off / val_size,3))
    on_diag_sum, off_diag_sum = 0,0

    #scheduler with warmup
    if epoch >= warmup:
        sch1.step()
        sch2.step()


    if val_loss < best_val_loss:
        torch.save({"hyper_parameters": mpnn_1.hparams, "state_dict": mpnn_1.state_dict()}, 'BT.ckpt')
        best_val_loss = val_loss

loss_curve = {"Train_loss":losses, "Val_loss":val_losses, "On_diag":on_diag_losses, "Off_diag":off_diag_losses}

plot_losses(losses, val_losses, 'Barlow_Twins')

df = pd.DataFrame(loss_curve)
df.to_csv('Barlow_Twins/loss_curve.csv')
