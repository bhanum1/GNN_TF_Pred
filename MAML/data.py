import pandas as pd
from chemprop import data, featurizers, models, nn
import random

def generate_data(csv, tasks):
    input_path = csv # path to your data .csv file
    df = pd.read_csv(input_path) #convert to dataframe
    

    #get columns
    smis = df['smiles']
    targets = df['target']
    task_labels = df['task']

    #create holders for all dataloaders
    task_data_dict = dict()
    for task in tasks:
        #Find correct section of data
        indices = task_labels == task
        task_smis = smis.loc[indices]
        task_targets = targets.loc[indices]

        #create data
        task_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(task_smis, task_targets)]
        task_data_dict[task] = task_data


    return task_data_dict

def get_loaders(task_data, m_support, k_query, test_indices):
    num_workers = 10 # number of workers for dataloader. 0 means using main process for data loading
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

