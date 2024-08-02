import pandas as pd
from rdkit import Chem
import random
from chemprop import data, featurizers, models, nn
import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

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

def subgraph(G, center, percent):
    num = int(np.ceil(len(G.nodes)*(1-percent)))
    visited = set()
    removed = []
    stack = [center]
    while stack:
        n = stack.pop(0)
        if n in visited:
            continue
        visited.add(n)
        a = G.neighbors(n)
        if len(visited) >= num:
            removed.append(n)
            G.remove_node(n)
        b = [x for x in a if x not in visited or x not in stack]
        stack.extend(b)

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


# Convert SMILES to RDKit Mol object
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)


# Clear aromaticity flags
def clear_aromaticity(mol):
    # Create a new molecule with aromaticity flags cleared
    emol = Chem.EditableMol(mol)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(False)
    return emol.GetMol()

# Recalculate aromaticity
def recalculate_aromaticity(mol):
    try:
        # Clear aromaticity and sanitize
        mol = clear_aromaticity(mol)
        Chem.SanitizeMol(mol, Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_CLEANUP)
        return mol
    except Exception as e:
        print(f"Sanitization error: {e}")
        return False

# Validate molecule
def validate_mol(mol):
    return recalculate_aromaticity(mol)

# Convert molecule to SMILES
def mol_to_smiles(mol):
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        print(f"SMILES conversion error: {e}")
        return None

# Augment SMILES by removing 1 random atom
def augment_smiles(smiles):
    mol = smiles_to_mol(smiles)
    N = mol.GetNumAtoms()
    G = mol_to_networkx_graph(mol)

    i = random.sample(list(range(N)), 1)

    if mol is not None:
        G_new, _ = subgraph(mol_to_networkx_graph(mol), i[0], 0.25)
        modified_mol = networkx_graph_to_mol(G_new)

        if modified_mol and validate_mol(modified_mol):
            return mol_to_smiles(validate_mol(modified_mol))
        else:
            return "Invalid molecule after modification."
    return "Invalid input molecule."

def create_loaders(original_smiles, batch_size = 64, num_workers = 0):
    smis_1, smis_2 = [], []
    for smiles in original_smiles:
        aug_smi_1 = augment_smiles(smiles)
        aug_smi_2 = augment_smiles(smiles)

        smis_1.append(aug_smi_1)
        smis_2.append(aug_smi_2)

    for _ in range(3):
        remove_duplicates(smis_1, smis_2, original_smiles)



    data_1, data_2 = [], []
    fail = 0
    for i in range(len(smis_1)):
        try:
            mol_1 = data.MoleculeDatapoint.from_smi(smis_1[i], 0)
            mol_2 = data.MoleculeDatapoint.from_smi(smis_2[i], 0)
            
            if mol_1 and mol_2:
                data_1.append(mol_1)
                data_2.append(mol_2)
        except:
            fail += 1

    #print("Percent failed: ", 100 * fail/len(data_1))

    indices = list(range(len(data_1)))
    random.shuffle(indices)
    num = round(len(indices) * 0.9)
    train_indices = indices[:num]
    val_indices = indices[num:]


    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_data, val_data, _ = data.split_data_by_indices(
        data_1, train_indices, val_indices, None
    )
    train_data_2, val_data_2, _ = data.split_data_by_indices(
        data_2, train_indices, val_indices, None
    )

    train_dset = data.MoleculeDataset(train_data, featurizer)
    val_dset = data.MoleculeDataset(val_data, featurizer)
    train_dset_2 = data.MoleculeDataset(train_data_2, featurizer)
    val_dset_2 = data.MoleculeDataset(val_data_2, featurizer)

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers, shuffle = False)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    train_loader_2 = data.build_dataloader(train_dset_2, batch_size=batch_size, num_workers=num_workers, shuffle = False)
    val_loader_2 = data.build_dataloader(val_dset_2, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, train_loader_2, val_loader_2

# Example usage
original_smiles = pd.read_csv('Barlow_Twins/dataset.csv')['smiles']

def remove_duplicates(smi1,smi2, smiles):
    for i in range(len(smi1)):
        if smi1[i] == smi2[i]:
            smi1[i] = augment_smiles(smiles[i])


'''
smi1, smi2 = [], []

for smiles in original_smiles:
    augmented_smiles = augment_smiles(smiles)
    aug2 = augment_smiles(smiles)
    smi1.append(augmented_smiles)
    smi2.append(aug2)

same = 0
for i in range(len(smi1)):
    if smi1[i] == smi2[i]:
        same += 1
print(same / len(smi1))

for _ in range(3):
    remove_duplicates(smi1, smi2, original_smiles)

    same = 0
    for i in range(len(smi1)):
        if smi1[i] == smi2[i]:
            same += 1
    print(same / len(smi1))
'''