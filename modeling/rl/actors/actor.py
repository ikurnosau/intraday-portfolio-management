from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor
from ...modeling_utils import smooth_abs


class FullyConnectedBackend(nn.Module):
    def __init__(self, 
        n_assets: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ): 
        super().__init__()

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

    def forward(self, features: torch.Tensor):
        h = self.fc_shared(features)  # (B, hidden_dim)
        v = torch.tanh(self.fc_out(h))  # (B, n_assets)
        return v


class TransformerBackend(nn.Module):
    def __init__(self,
        d_model: int=64, 
        nhead: int=4,
        num_layers: int=1,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        
        input_proj_layers = [nn.Linear(3, d_model)]
        if use_layer_norm:
            input_proj_layers.append(nn.LayerNorm(d_model))
        input_proj_layers.append(nn.ReLU())
        if dropout > 0.0:
            input_proj_layers.append(nn.Dropout(dropout))
        self.input_proj = nn.Sequential(*input_proj_layers)

        # Transformer encoder block with dropout and norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # more stable in practice
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: produces signed scalar per position in [-1, 1]
        self.output_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, features: torch.Tensor):
        signal, position, vol_to_spread = torch.chunk(features, chunks=3, dim=1)

        x = torch.stack([signal, position, vol_to_spread], dim=2)
        x = self.input_proj(x)  # (batch, n_assets, d_model)
        x = self.encoder(x)     # (batch, n_assets, d_model)
        score = torch.tanh(self.output_head(x).squeeze(-1))  # (batch, n_assets)

        return score


class RlActor(nn.Module, BaseActor):
    """Policy network mapping :class:`State` → portfolio allocation.

    The allocation vector ``a`` (*) satisfies::

        -1 ≤ aᵢ ≤ 1  and  Σ |aᵢ| = 1
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        backend: nn.Module,
        n_assets: int,
        train_signal_predictor: bool = False,
        exploration_eps: float = 0.05,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.backend = backend
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
            torch.full((n_assets,), 4.29),    # vol/spread
        ]))
        self.register_buffer("sigma", torch.cat([
            torch.full((n_assets,), 0.05),
            torch.full((n_assets,), 0.577),   # √(1/3)
            torch.full((n_assets,), 4.59),
        ]))

    def forward(self, state: State):
        if not self.train_signal_predictor:
            with torch.no_grad():
                signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)
        else:
            signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)

        features = torch.cat([signal_repr, state.position, state.volatility / state.spread], dim=-1)
        features = (features - self.mu) / (self.sigma + 1e-8)
        v = self.backend(features)  # (B, n_assets)

        if self.training and self.exploration_eps > 0.0:
            noise = self.exploration_eps * torch.randn_like(v)
            v = v + noise

        action = v / (smooth_abs(v).sum(dim=-1, keepdim=True) + 1e-8)

        return action

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)

        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self
