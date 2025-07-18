from __future__ import annotations

from typing import List

import torch
from torch.optim import Adam
import logging

from ..agent import RlAgent


class PolicyGradient:
    """Vanilla REINFORCE implementation for discrete action spaces."""

    def __init__(self, agent: RlAgent, lr: float = 1e-3, gamma: float = 0.99, device: torch.device | str = "cuda"):
        self.agent = agent
        self.gamma = gamma
        self.device = torch.device(device)

        # Only update *learnable* parameters of the actor (predictor params are frozen)
        self.optimizer = Adam(
            [p for p in self.agent.actor.parameters() if p.requires_grad], lr=lr
        )

    @staticmethod
    def _discount_rewards(rewards: List[torch.Tensor], gamma: float) -> torch.Tensor:
        """Compute discounted returns *in-place* to avoid additional memory."""
        discounted = []
        R = torch.tensor(0.0, dtype=torch.float32, device=rewards[0].device)
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)
        return torch.stack(discounted)

    def train(self, epochs: int = 1):
        trading_days = self.agent.get_trading_days()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for day in trading_days:
                trajectory = self.agent.generate_trajectory(day)
                if not trajectory:
                    continue

                # We treat actor as deterministic; use direct gradient through reward.
                rewards = [step[2] for step in trajectory]  # list of scalar tensors

                rewards_t = torch.stack(rewards)  # (T,)

                eps = 1e-6
                loss = -torch.log(torch.clamp(1.0 + rewards_t, min=eps)).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logging.info(f"loss: {loss.item()}, rewards_t: {rewards_t.mean().item()}")

                epoch_loss += loss.item()

            print(f"[PolicyGradient] Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss:.4f}")
