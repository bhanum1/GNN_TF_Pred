import pandas as pd
from chemprop import data, featurizers, models, nn

import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy


# from torch.utils.data import Dataset, DataLoader
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

num_workers = 0


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

def generate_data(csv):
    input_path = csv # path to your data .csv file
    df = pd.read_csv(input_path) #convert to dataframe
    
    #get columns
    smis = df['smiles']
    targets = df['target']

    #create data
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, targets)]
    mols = [d.mol for d in all_data]  # RDkit Mol objects are use for structure based splits
    
    smis_1, smis_2 = [],[]

    for i in range(len(mols)):
        molGraph = mol_to_networkx_graph(mols[i])

        start_i, start_j = random.sample(list(range(mols[i].GetNumAtoms())), 2)

        percent_i, percent_j = 0.15, 0.15
        G_i, removed_i = removeSubgraph(molGraph, start_i, percent_i)
        G_j, removed_j = removeSubgraph(molGraph, start_j, percent_j)
        
        mol_1, mol_2 = networkx_graph_to_mol(G_i), networkx_graph_to_mol(G_j)

        mol_1.ClearComputedProps()
        mol_2.ClearComputedProps()
        
        smi_a = Chem.MolToSmiles(mol_1)
        smi_b = Chem.MolToSmiles(mol_2)

        smis_1.append(smi_a)
        smis_2.append(smi_b)

    data_1 = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_1, targets)]
    data_2 = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_2, targets)]

    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))

    train_data, val_data, test_data = data.split_data_by_indices(
        data_1, train_indices, val_indices, test_indices
    )
    train_data_2, val_data_2, test_data_2 = data.split_data_by_indices(
        data_2, train_indices, val_indices, test_indices
    )

    train_dset = data.MoleculeDataset(train_data)
    val_dset = data.MoleculeDataset(val_data)
    test_dset = data.MoleculeDataset(test_data)

    train_dset_2 = data.MoleculeDataset(train_data_2)
    val_dset_2 = data.MoleculeDataset(val_data_2)
    test_dset_2 = data.MoleculeDataset(test_data_2)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle = False)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)
    train_loader_2 = data.build_dataloader(train_dset_2, num_workers=num_workers, shuffle = False)
    val_loader_2 = data.build_dataloader(val_dset_2, num_workers=num_workers, shuffle=False)
    test_loader_2 = data.build_dataloader(test_dset_2, num_workers=num_workers, shuffle=False)

    return [train_loader, val_loader, test_loader, train_loader_2, val_loader_2, test_loader_2]



loaders = generate_data('Barlow_Twins/dataset.csv')
train_loader = loaders[0]

for batch in train_loader:
    bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch

    print(bmg)


