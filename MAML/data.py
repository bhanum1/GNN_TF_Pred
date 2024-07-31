import pandas as pd
from chemprop import data, featurizers, models, nn
import random


import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger           


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def generate_data(csv, tasks, test=False):
    input_path = csv # path to your data .csv file
    df = pd.read_csv(input_path) #convert to dataframe
    

    #get columns
    smis = df['smiles']
    targets = df['target']
    task_labels = df['task']

    #create holders for all dataloaders
    task_data_dict = dict()
    val_data_dict  = dict()
    for task in tasks:
        #Find correct section of data
        indices = task_labels == task
        task_smis = smis.loc[indices]
        task_targets = targets.loc[indices]

        #create data
        task_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis, task_targets)]
        
        #split into train/val
        if not test:
            indices = list(range(len(task_smis)))
            random.shuffle(indices)
            
            num = round(len(indices) * 0.8)
            train_indices = indices[:num]
            val_indices = indices[num:]

            
        else:
            train_indices = list(range(len(task_smis)))
            val_indices = None

        train_data, val_data, _ = data.split_data_by_indices(task_data, train_indices, val_indices,None)
        task_data_dict[task] = train_data
        val_data_dict[task] = val_data


    return task_data_dict, val_data_dict

def get_loaders(task_data, m_support, k_query, test_indices):
    num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    
    #choose m support and k query at random, no overlap
    indices=list(range(len(task_data)))
    random.shuffle(indices)

    if not test_indices:
        supp_indices = indices[:m_support]
        query_indices = indices[m_support:m_support+k_query]

        #create datasets
        supp_data, query_data, _ = data.split_data_by_indices(task_data, supp_indices, query_indices,None)
        test_loader = None

    else:
        train_indices = []
        for i in indices:
            if i not in test_indices:
                train_indices.append(i)
        
        supp_indices = train_indices[:m_support]
        query_indices = train_indices[m_support:m_support+k_query]
        supp_data, query_data, test_data = data.split_data_by_indices(task_data, supp_indices, query_indices, test_indices)

        test_dataset = data.MoleculeDataset(test_data, featurizer)
        test_loader = data.build_dataloader(test_dataset, num_workers=num_workers,batch_size = 50)

    supp_dataset = data.MoleculeDataset(supp_data, featurizer)
    query_dataset = data.MoleculeDataset(query_data, featurizer)

    supp_loader = data.build_dataloader(supp_dataset, num_workers=num_workers,batch_size=m_support)
    query_loader = data.build_dataloader(query_dataset, num_workers=num_workers,batch_size=k_query)

    return supp_loader, query_loader, test_loader

