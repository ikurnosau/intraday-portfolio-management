from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List

import pandas as pd
import torch
import logging
import numpy as np

from data.processed.dataset_creation import DatasetCreator
from .state import State
from .reward import estimated_return_reward


class PortfolioEnvironment:
    """Minimal episodic environment iterating through a single trading day."""

    def __init__(
        self,
        transaction_fee: float = 0.0005,
        reward_function: Callable[[State, State, float], torch.Tensor] | None = None,
    ):
        self.transaction_fee = transaction_fee
        self.reward_function = reward_function or estimated_return_reward

        self.signal_features_trajectory_batch = None
        self.next_returns_trajectory_batch = None
        self.spreads_trajectory_batch = None
        self.state_index = None
        self.current_state = None

    def _prepare_state_template(self, i: int) -> State:
        return State(
            signal_features=self.signal_features_trajectory_batch[:, i, :, :, :],
            next_step_return=self.next_returns_trajectory_batch[:, i, :],
            spread=self.spreads_trajectory_batch[:, i, :],
            position=torch.zeros(self.spreads_trajectory_batch[:, i, :].shape),
        )

    def reset(self,
        signal_features_trajectory_batch: torch.Tensor,
        next_returns_trajectory_batch: torch.Tensor,
        spreads_trajectory_batch: torch.Tensor,
    ) -> State:
        """
        signal_features_trajectory_batch: (batch_size, trajectory_length, n_assets, seq_len,n_features)
        next_returns_trajectory_batch: (batch_size, trajectory_length, n_assets)
        spreads_trajectory_batch: (batch_size, trajectory_length, n_assets)
        """

        self.signal_features_trajectory_batch = signal_features_trajectory_batch
        self.next_returns_trajectory_batch = next_returns_trajectory_batch
        self.spreads_trajectory_batch = spreads_trajectory_batch
        self.state_index = 0
        self.current_state = self._prepare_state_template(self.state_index)

        return self.current_state

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, State | None]:
        next_index = self.state_index + 1
        if next_index >= self.signal_features_trajectory_batch.shape[1]:
            # End of episode
            return torch.tensor(np.zeros(len(self.signal_features_trajectory_batch))), None

        next_state = self._prepare_state_template(next_index)
        next_state.position = action

        reward = self.reward_function(self.current_state, next_state, self.transaction_fee)

        self.state_index = next_index
        self.current_state = next_state

        return reward, next_state
