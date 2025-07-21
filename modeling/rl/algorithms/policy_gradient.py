from __future__ import annotations

from typing import List

import torch
from torch.optim import Adam
import logging
from tqdm import tqdm

from ..agent import RlAgent
from ..state import State

class PolicyGradient:
    def __init__(self, 
        agent: RlAgent,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        lr: float = 1e-3,
        device: torch.device | str = "cuda",
    ):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)

        # Only update *learnable* parameters of the actor (predictor params are frozen)
        self.optimizer = Adam(
            [p for p in self.agent.actor.parameters() if p.requires_grad], lr=lr
        )

    @staticmethod
    def _loss(rewards: list[torch.Tensor]) -> torch.Tensor:
        # Shape → (T, batch_size)
        rewards_t = torch.stack(rewards)
        
        # Cumulative multiplicative return per sample: Π_t (1 + r_t)
        cumulative_return = torch.clamp(1.0 + rewards_t, min=1e-6).prod(dim=0)  # (batch_size,)

        # Maximise cumulative_return ⇔ minimise negative log of it
        return -torch.log(cumulative_return).mean() 

    def train(self, epochs: int = 1):
        for epoch in range(epochs):
            epoch_loss = 0.0
            realized_returns = []
            for signal_features_trajectory_batch, next_returns_trajectory_batch, spreads_trajectory_batch in tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                leave=False,
            ):
                trajectory = self.agent.generate_trajectory(
                    signal_features_trajectory_batch=signal_features_trajectory_batch.to(self.device, non_blocking=True),
                    next_returns_trajectory_batch=next_returns_trajectory_batch.to(self.device, non_blocking=True),
                    spreads_trajectory_batch=spreads_trajectory_batch.to(self.device, non_blocking=True),
                )
                if not trajectory:
                    continue

                rewards = [step[2] for step in trajectory]  # list[(batch_size,)] length = T
                realized_returns += rewards

                loss = self._loss(rewards)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"[PolicyGradient] Epoch {epoch + 1}/{epochs} — Loss: {epoch_loss:.4f}")
