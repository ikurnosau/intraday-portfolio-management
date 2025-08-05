from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor
from ...modeling_utils import smooth_abs


class RlActor(nn.Module, BaseActor):
    """Policy network mapping :class:`State` → portfolio allocation.

    The allocation vector ``a`` (*) satisfies::

        -1 ≤ aᵢ ≤ 1  and  Σ |aᵢ| = 1
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
        exploration_eps: float = 0.05,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.n_assets = n_assets
        self.exploration_eps = exploration_eps

        # Freeze predictor parameters (assumed pre-trained)
        self.train_signal_predictor = train_signal_predictor
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

        self.register_buffer("mu",  torch.cat([
            torch.full((n_assets,), 0.50),    # predictor
            torch.zeros(n_assets),            # position
            torch.full((n_assets,), 3e-4),    # spread
        ]))
        self.register_buffer("sigma", torch.cat([
            torch.full((n_assets,), 0.05),
            torch.full((n_assets,), 0.577),   # √(1/3)
            torch.full((n_assets,), 3e-4),
        ]))

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

        features = torch.cat([signal_repr, state.position, state.spread], dim=-1)
        features = (features - self.mu) / (self.sigma + 1e-8)
        h = self.fc_shared(features)  # (B, hidden_dim)

        v = torch.tanh(self.fc_out(h))  # (B, n_assets)

        action =  v / (smooth_abs(v).sum(dim=-1, keepdim=True) + 1e-8)

        # --- Single-stock exploration jitter ---------------------------------
        if self.training and self.exploration_eps > 0.0:
            B, A = action.shape
            idx = torch.randint(low=0, high=A, size=(B,), device=action.device)
            jitter = torch.zeros_like(action)
            jitter[torch.arange(B), idx] = self.exploration_eps * torch.randn(B, device=action.device)
            noisy = action + jitter
            action = noisy / (smooth_abs(noisy).sum(dim=-1, keepdim=True) + 1e-8)

        return action

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)

        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self
