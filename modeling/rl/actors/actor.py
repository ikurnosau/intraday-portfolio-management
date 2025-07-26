from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class RlActor(nn.Module, BaseActor):
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
        train_signal_predictor: bool = False,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.n_assets = n_assets

        # Freeze predictor parameters (assumed pre-trained)
        self.train_signal_predictor = train_signal_predictor
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

        # --- Build a deeper shared MLP backbone -------------------------------------------------
        # The network depth, dropout probability and use of LayerNorm can be configured via
        # constructor arguments. This makes the actor more flexible while preserving backward
        # compatibility with previous checkpoints that relied on the single-layer layout.

        layers: list[nn.Module] = []

        in_features = n_assets * 3  # predictor output + position + spread
        for i in range(num_layers):
            layers.append(nn.Linear(in_features if i == 0 else hidden_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.fc_shared = nn.Sequential(*layers)

        # Final projection to the n_assets allocation vector
        self.fc_out = nn.Linear(hidden_dim, n_assets)


    def forward(self, state: State):
        if not self.train_signal_predictor:
            with torch.no_grad():
                signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)
        else:
            signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)

        h = self.fc_shared(
            torch.cat([signal_repr, state.position, state.spread], dim=-1)
        )  # (B, hidden_dim)

        v = torch.tanh(self.fc_out(h))  # (B, n_assets)

        action = v / (v.abs().sum(dim=-1, keepdim=True) + 1e-8)  # (B, n_assets)

        return action
