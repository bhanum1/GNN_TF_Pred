from chemprop import data, featurizers, models, nn
import torch
import torch.nn.functional as F
from data_processing import *
import torch.optim as optim
from BT_loss import BT_loss
from rdkit import RDLogger
from plots import plot_losses
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dropout = 0
mp = nn.BondMessagePassing(depth=3, dropout=dropout, d_h = 256, activation = 'leakyrelu')
ffn = nn.RegressionFFN(dropout=dropout, input_dim = 256, hidden_dim = 256) # regression head
agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
batch_norm = False

#initialize the model
mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric()])



opt1 = optim.SGD(mpnn.parameters(), lr = 0.0005)


input_path = 'Barlow_Twins/dataset.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe



from chemprop.cli.utils.parsing import build_data_from_files, make_dataset
from pathlib import Path

batch_size = 64
num_workers = 0

input_path = '/home/bhanu/Documents/GitHub/Thermal_Fluid_Prediction_GNN/Datasets/viscosity_data.csv' # path to your data .csv file
df = pd.read_csv(input_path) #convert to dataframe

smis = df['smiles']
targets = df['Viscosity']
temps = df['temperature']
lnA_targets = df['lnA']

all_data = [data.MoleculeDatapoint.from_smi(smi, y, temp=temp, lnA_target=lnA) for smi, y,temp,lnA in zip(smis, targets,temps,lnA_targets)]
mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits

grouped = df.groupby(df['splits'].str.lower())
train_indices = grouped.groups.get("train", pd.Index([])).tolist()
val_indices = grouped.groups.get("val", pd.Index([])).tolist()
test_indices = grouped.groups.get("test", pd.Index([])).tolist()
train_indices, val_indices, test_indices = [train_indices], [val_indices], [test_indices]

train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

train_data = train_data[0]
val_data = val_data[0]
test_data = test_data[0]

train_dset = make_dataset(train_data, reaction_mode='REAC_PROD')
val_dset = make_dataset(val_data,reaction_mode='REAC_PROD')
test_dset = make_dataset(test_data, reaction_mode='REAC_PROD')

train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle = False)
val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


best_val_loss = float('inf')
criterion = torch.nn.MSELoss(reduction='mean')
epochs = 100

for epoch in range(epochs):

    train_loss = 0
    for batch in train_loader:
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask, temps, lnA_targets, EaR_targets = batch

        pred = mpnn(bmg)
        l = criterion(pred, targets, temps = temps, lnA_targets = lnA_targets)

        train_loss += l.item()

    print(train_loss)
