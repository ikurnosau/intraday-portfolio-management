from __future__ import annotations

import torch
import logging

from .environment import PortfolioEnvironment
from .state import State
from .actors.base_actor import BaseActor
from .actors.actor import RlActor
from time import perf_counter
from contextlib import contextmanager


class RlAgent:
    """Couples a policy network with the trading environment."""

    def __init__(self, actor: BaseActor, env: PortfolioEnvironment, single_action_per_trajectory: bool = False):
        self.actor = actor
        self.env = env

        self.single_action_per_trajectory = single_action_per_trajectory
        self.last_action = None

        self.current_state: State | None = None

    def step(self) -> tuple[State, torch.Tensor, torch.Tensor] | None:
        if self.single_action_per_trajectory: 
            if self.last_action is None:
                action = self.actor(self.current_state)
                self.last_action = action
            else:
                action = self.last_action
        else: 
            action = self.actor(self.current_state)

        reward, next_state = self.env.step(action)

        if next_state is None:
            # Episode finished
            return None

        prev_state = self.current_state
        self.current_state = next_state
        return prev_state, action, reward

    def generate_trajectory(self,
        signal_features_trajectory_batch: torch.Tensor,
        next_returns_trajectory_batch: torch.Tensor,
        spreads_trajectory_batch: torch.Tensor,
        volatility_trajectory_batch: torch.Tensor,
    ):
        self.current_state = self.env.reset(
            signal_features_trajectory_batch=signal_features_trajectory_batch,
            next_returns_trajectory_batch=next_returns_trajectory_batch,
            spreads_trajectory_batch=spreads_trajectory_batch,
            volatility_trajectory_batch=volatility_trajectory_batch,
        )

        if self.single_action_per_trajectory:
            self.last_action = None

        trajectory = []
        while True:
            step_out = self.step()
            if step_out is None:
                break

            trajectory.append(step_out)

        return trajectory 