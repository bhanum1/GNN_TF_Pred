#import packages (can do pip install chemprop)
import pandas as pd
from chemprop import data, featurizers, models, nn

from lightning import pytorch as pl
from model import *
from lightning.pytorch.callbacks import ModelCheckpoint


datafile = "/home/bhanu/Documents/GitHub/chemprop/tests/data/regression/mol/mol.csv"
num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
smiles_column = 'smiles' # name of the column containing SMILES strings
target_columns = ['lipo'] # list of names of the columns containing targets

# gpu stuff
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize the model
mpnn = build_model(0.3)
mpnn.to(device)
criterion = torch.nn.MSELoss(reduction='mean')

#get columns
df_input = pd.read_csv(datafile)
smis = df_input.loc[:, smiles_column].values
ys = df_input.loc[:, target_columns].values

all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(train_data, featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(val_data, featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(test_data, featurizer)
test_dset.normalize_targets(scaler)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)

checkpointing = ModelCheckpoint(
            "checkpoints",
            "best-{epoch}-{val_loss:.2f}",
            "val_loss",
            mode="min",
            save_last=True,
        )



trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    callbacks=[checkpointing],
    max_epochs=500, # number of epochs to train for
)

trainer.fit(mpnn, train_loader, val_loader)


results = trainer.test(mpnn, test_loader)