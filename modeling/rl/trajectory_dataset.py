import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal


class TrajectoryDataset(Dataset):
    def __init__(self, 
                 signal_features: np.ndarray, 
                 next_returns: np.ndarray, 
                 spreads: np.ndarray, 
                 volatility: np.ndarray, 
                 trajectory_length: int, 
                 horizon: int, 
                 include_interhorizon_trajectories: bool=False):
        signal_features_list, next_returns_list, spreads_list, volatility_list = [], [], [], []
        start_offsets = np.arange(0, horizon) if include_interhorizon_trajectories else [0]
        for start_offset in start_offsets:
            for block_start in range(start_offset, len(signal_features) - horizon + 1, horizon): 
                signal_features_list.append(signal_features[block_start])
                next_returns_list.append(np.cumprod(next_returns[block_start:block_start + horizon] + 1, axis=0) - 1)
                spreads_list.append(spreads[block_start])
                volatility_list.append(volatility[block_start])

            signal_features_list = signal_features_list[:len(signal_features_list) - (len(signal_features_list) % trajectory_length)]
            next_returns_list = next_returns_list[:len(next_returns_list) - (len(next_returns_list) % trajectory_length)]
            spreads_list = spreads_list[:len(spreads_list) - (len(spreads_list) % trajectory_length)]
            volatility_list = volatility_list[:len(volatility_list) - (len(volatility_list) % trajectory_length)]

        self.signal_features = torch.tensor(np.array(signal_features_list), dtype=torch.float32)
        self.next_returns = torch.tensor(np.array(next_returns_list), dtype=torch.float32)
        self.spreads = torch.tensor(np.array(spreads_list), dtype=torch.float32)
        self.volatility = torch.tensor(np.array(volatility_list), dtype=torch.float32)
        self.trajectory_length = trajectory_length

    def __len__(self):
        return len(self.signal_features) // self.trajectory_length

    def __getitem__(self, idx):
        start_idx = idx * self.trajectory_length
        end_idx = start_idx + self.trajectory_length
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