from __future__ import annotations

import torch
import torch.nn as nn
import logging

from ..state import State


class RlActor(nn.Module):
    """Policy network mapping :class:`State` → portfolio allocation.

    The allocation vector ``a`` (*) satisfies::

        -1 ≤ aᵢ ≤ 1  and  Σ |aᵢ| = 1

    It is produced by sampling a mixture of

    1. **Magnitude** – drawn from a Dirichlet distribution so the positive
       components sum to 1.
    2. **Sign**      – independent Bernoulli variables mapping {0,1} → {+1,−1}.

    The method returns a single tensor ``action``.
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        n_assets: int,
        hidden_dim: int = 128,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.n_assets = n_assets

        # Freeze predictor parameters (assumed pre-trained)
        for p in self.signal_predictor.parameters():
            p.requires_grad = False
        self.signal_predictor.eval()

        self.fc_shared = nn.Sequential(
            nn.Linear(n_assets * 3, hidden_dim),
            nn.ReLU(),
        )
        self.fc_out = nn.Linear(hidden_dim, n_assets)

        self.device = torch.device(device)

    def forward(self, state: State):
        state.signal_features = state.signal_features.to(self.device, non_blocking=True)
        state.position = state.position.to(self.device, non_blocking=True)
        state.spread = state.spread.to(self.device, non_blocking=True)

        with torch.no_grad():
            signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)

        h = self.fc_shared(
            torch.cat([signal_repr, state.position, state.spread], dim=-1)
        )  # (B, hidden_dim)

        v = torch.tanh(self.fc_out(h))  # (B, n_assets)

        action = v / (v.abs().sum(dim=-1, keepdim=True) + 1e-8)  # (B, n_assets)

        return action
