from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..state import State


class RlActor(nn.Module):
    """Policy network mapping :class:`State` → categorical action distribution."""

    def __init__(self, signal_predictor: nn.Module, hidden_dim: int = 64, n_actions: int = 3):
        super().__init__()
        self.signal_predictor = signal_predictor

        # Freeze the predictor – assumed to be pre-trained.
        for p in self.signal_predictor.parameters():
            p.requires_grad = False
        self.signal_predictor.eval()

        # Infer predictor output dimensionality.
        if hasattr(signal_predictor, "n_class"):
            predictor_dim = signal_predictor.n_class
        else:
            predictor_dim = next(reversed(self.signal_predictor.parameters())).shape[0]

        self.fc1 = nn.Linear(predictor_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state: State) -> Categorical:
        # Signal predictor expects a batch dimension.
        x = state.signal_features.unsqueeze(0)
        with torch.no_grad():
            signal_logits = self.signal_predictor(x).squeeze(0)  # (predictor_dim,)

        extra = torch.stack((state.position, state.spread))  # (2,)
        h = torch.relu(self.fc1(torch.cat([signal_logits, extra], dim=-1)))
        logits = self.fc2(h)
        return Categorical(logits=logits)
