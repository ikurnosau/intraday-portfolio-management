import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal


class TrajectoryDataset(Dataset):
    def __init__(self, signal_features: np.ndarray, next_returns: np.ndarray, spreads: np.ndarray, trajectory_length: int):
        self.signal_features = torch.tensor(signal_features, dtype=torch.float32)
        self.next_returns = torch.tensor(next_returns, dtype=torch.float32)
        self.spreads = torch.tensor(spreads, dtype=torch.float32)
        self.trajectory_length = trajectory_length

    def __len__(self):
        return len(self.signal_features) // self.trajectory_length

    def __getitem__(self, idx):
        start_idx = idx * self.trajectory_length
        end_idx = start_idx + self.trajectory_length
        return self.signal_features[start_idx:end_idx], self.next_returns[start_idx:end_idx], self.spreads[start_idx:end_idx]
    
    def as_dataloader(self, batch_size: int=8, shuffle: bool=True, num_workers=8, prefetch_factor=4, pin_memory=True, persistent_workers=True, drop_last=True): 
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last)