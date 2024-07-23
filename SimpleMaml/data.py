import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class SinusoidDataset(Dataset):

    def __init__(self, task, size):
        self.amplitude, self.phase = task

        x = np.random.uniform(0, math.pi * 4, size)

        y = []
        for point in x:
            y_point = self.amplitude * math.sin(point + self.phase)
            y.append(y_point)
        
        self.x_data = torch.tensor(x, dtype=torch.float32).reshape(-1,1)
        self.y_data = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    


def get_loaders(task, m_support, k_query):
    num_workers = 0 # number of workers for dataloader. 0 means using main process for data loading
    
    s_dataset, q_dataset = SinusoidDataset(task,m_support), SinusoidDataset(task,k_query)

    s_loader = DataLoader(s_dataset, batch_size = m_support)
    q_loader = DataLoader(q_dataset, batch_size = k_query)

    

    return s_loader, q_loader

