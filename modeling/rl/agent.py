from __future__ import annotations

import torch

from .environment import PortfolioEnvironment
from .state import State
from .actors.actor import RlActor


class RlAgent:
    """Couples a policy network with the trading environment."""

    def __init__(self, actor: RlActor, env: PortfolioEnvironment, device: torch.device | str = "cuda"):
        self.actor = actor.to(device)
        self.env = env
        self.device = torch.device(device)
        self.current_state: State | None = None

    def set_trading_day(self, day):
        """Reset environment to *day* and obtain the initial state."""
        self.current_state = self.env.set_trading_day(day)

    def get_trading_days(self):
        return self.env.get_trading_days()

    def step(self):
        if self.current_state is None:
            raise RuntimeError("Trading day not initialised; call set_trading_day first.")

        action = self.actor(self.current_state)

        reward, next_state = self.env.step(action.to(self.device))
        if next_state is None:
            # Episode finished
            return None

        prev_state = self.current_state  # keep gradient flow through position
        self.current_state = next_state
        return prev_state, action, reward

    def generate_trajectory(self, day):
        """Collect a full-day trajectory of (state, action, log_prob, reward) tuples."""
        self.set_trading_day(day)
        trajectory = []
        while True:
            step_out = self.step()
            if step_out is None:
                break
            trajectory.append(step_out)
        return trajectory 