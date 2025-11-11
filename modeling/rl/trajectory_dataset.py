import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from typing import Literal


class TrajectoryDataset(Dataset):
    def __init__(self, signal_features: np.ndarray, next_returns: np.ndarray, spreads: np.ndarray, volatility: np.ndarray, trajectory_length: int, horizon: int, shift_data_within_horizon: bool=False):
        signal_features_list, next_returns_list, spreads_list, volatility_list = [], [], [], []
        start_offsets = np.arange(0, horizon) if shift_data_within_horizon else [0] 
        for start_offset in start_offsets: 
            signal_features_list.extend(signal_features[start_offset::horizon])
            next_returns_list.extend(next_returns[start_offset::horizon])
            spreads_list.extend(spreads[start_offset::horizon])
            volatility_list.extend(volatility[start_offset::horizon])
        
        self.signal_features = torch.tensor(np.array(signal_features_list), dtype=torch.float32)
        self.next_returns = torch.tensor(np.array(next_returns_list), dtype=torch.float32)
        self.spreads = torch.tensor(np.array(spreads_list), dtype=torch.float32)
        self.volatility = torch.tensor(np.array(volatility_list), dtype=torch.float32)
        self.trajectory_length = trajectory_length

    def __len__(self):
        return math.ceil(len(self.signal_features) / self.trajectory_length)

    def __getitem__(self, idx):
        start_idx = idx * self.trajectory_length
        end_idx = min(start_idx + self.trajectory_length, len(self.signal_features))
        return self.signal_features[start_idx:end_idx], self.next_returns[start_idx:end_idx], self.spreads[start_idx:end_idx], self.volatility[start_idx:end_idx]
    
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