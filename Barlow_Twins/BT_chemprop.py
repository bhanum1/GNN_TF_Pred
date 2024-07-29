import pandas as pd
from chemprop import data, featurizers, models, nn
import random


import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed

def networkx_graph_to_mol(graph):
    mol = Chem.RWMol()
    
    atom_map = {}
    for node, attrs in graph.nodes(data=True):
        #print(node, attrs)
        atom_type = attrs.get('atom_type', 1)
        atom = Chem.Atom(atom_type)
        chirality = attrs.get('chirality', Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        atom.SetChiralTag(chirality)
        atom_idx = mol.AddAtom(atom)
        atom_map[node] = atom_idx
    
    for u, v, attrs in graph.edges(data=True):
        #print(u, v, attrs)
        bond_type = attrs.get('bond_type', 0)
        #print(bond_type_idx)
        bond_type_idx = BOND_LIST.index(bond_type)
        bond_dir = attrs.get('bond_dir', Chem.rdchem.BondDir.NONE)
        mol.AddBond(atom_map[u], atom_map[v], bond_type)
    
    return mol.GetMol()


def mol_to_networkx_graph(mol):
    G = nx.Graph()
    
    # Add nodes with attributes
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_type = atom.GetAtomicNum()
        chirality = atom.GetChiralTag()
        G.add_node(atom_idx, atom_type=atom_type, chirality=chirality)
    
    # Add edges with attributes
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_dir = bond.GetBondDir()
        G.add_edge(start_idx, end_idx, bond_type=bond_type, bond_dir=bond_dir)
    
    return G


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
        task_data_2 = deepcopy(task_data)
        
        for i in len(task_data):
            molGraph = mol_to_networkx_graph(task_data[i].mol)

            start_i, start_j = random.sample(list(range(d.mol.GetNumAtoms())), 2)

            percent_i, percent_j = 0.15, 0.15
            G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
            G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)

            task_data[i].mol = networkx_graph_to_mol(G_i)
            task_data_2[i].mol = networkx_graph_to_mol(G_j)




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

