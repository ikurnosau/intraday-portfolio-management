from __future__ import annotations

import torch
import torch.nn as nn

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
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.n_assets = n_assets

        # Freeze predictor parameters (assumed pre-trained)
        for p in self.signal_predictor.parameters():
            p.requires_grad = False
        self.signal_predictor.eval()

        # Determine predictor output dimension
        if hasattr(signal_predictor, "n_class"):
            predictor_dim = signal_predictor.n_class
        else:
            predictor_dim = next(iter(self.signal_predictor.parameters())).shape[0]

        self.fc_shared = nn.Sequential(
            nn.Linear(predictor_dim + 2, hidden_dim),
            nn.ReLU(),
        )

        # Single head producing raw allocations
        self.fc_out = nn.Linear(hidden_dim, n_assets)

    # ------------------------------------------------------------------ #
    # Forward – returns (action, log_prob)
    # ------------------------------------------------------------------ #

    def forward(self, state: State):
        # 1) Build feature vector
        x = state.signal_features.unsqueeze(0)  # (1, feat)
        with torch.no_grad():
            signal_repr = self.signal_predictor(x).squeeze(0)  # (feat',)

        extra = torch.stack((state.position, state.spread))  # (2,)
        h = self.fc_shared(torch.cat([signal_repr, extra], dim=-1))  # (hidden,)

        # 2) Raw weights in (-1,1)
        v = torch.tanh(self.fc_out(h))  # (n_assets,)

        # 3) ℓ¹-normalise so Σ|a| = 1
        action = v / (v.abs().sum() + 1e-8)

        return action
