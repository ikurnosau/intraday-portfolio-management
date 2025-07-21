from __future__ import annotations

import torch
import logging

from .environment import PortfolioEnvironment
from .state import State
from .actors.actor import RlActor
from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timer(label="Block"):
    start = perf_counter()
    yield
    end = perf_counter()
    logging.info(f"{label} took {end - start:.4f} seconds")


class RlAgent:
    """Couples a policy network with the trading environment."""

    def __init__(self, actor: RlActor, env: PortfolioEnvironment, device: torch.device | str = "cuda"):
        self.actor = actor.to(device)
        self.env = env

        self.current_state: State | None = None

    def step(self) -> tuple[State, torch.Tensor, torch.Tensor] | None:
        # with timer("Actor forward"):
        action = self.actor(self.current_state)

        # with timer("Environment step"):
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
    ):
        self.current_state = self.env.reset(
            signal_features_trajectory_batch=signal_features_trajectory_batch,
            next_returns_trajectory_batch=next_returns_trajectory_batch,
            spreads_trajectory_batch=spreads_trajectory_batch,
        )

        trajectory = []
        while True:
            step_out = self.step()
            if step_out is None:
                break

            trajectory.append(step_out)

        return trajectory 