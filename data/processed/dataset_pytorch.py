import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal


class DatasetPytorch(Dataset):
    def __init__(self, X, y, learning_task: Literal['classification', 'regression']):
        self.X = torch.tensor(X, dtype=torch.float32)
        if learning_task == 'classification':
            self.y = torch.tensor(y, dtype=torch.long) 
        elif learning_task == 'regression':
            self.y = torch.tensor(y, dtype=torch.float32) 
        else: 
            raise ValueError(f'Invalid learning task: {learning_task}')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def as_dataloader(self, batch_size: int=32, shuffle: bool=False): 
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)