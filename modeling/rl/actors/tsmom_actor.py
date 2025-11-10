from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class TSMomActor(nn.Module, BaseActor):
    """
    Time-series momentum actor.

    For each asset we compute the cumulative log return over the given
    *look_back_window*.  We go **long** if the momentum is positive and
    **short** if the momentum is negative.  All selected positions are
    equally-weighted such that the allocation vector *a* satisfies

        Σ |aᵢ| = 1, aᵢ ∈ [-1, 1].

    Assets with zero momentum receive zero allocation.
    """

    def __init__(self, look_back_window: int = 20):
        super().__init__()
        self.look_back_window = look_back_window

    def forward(self, state: State):
        # Expecting `signal_features` of shape (B, n_assets, seq_len, n_features)
        # Log-returns are stored in feature index 0.
        log_returns = state.signal_features[..., 0][..., -self.look_back_window:]  # (B, n_assets, L)

        # Time-series momentum score: cumulative log return per asset
        momentum = log_returns.sum(dim=-1)  # (B, n_assets)

        pos_mask = momentum > 0
        neg_mask = momentum < 0

        n_pos = pos_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        n_neg = neg_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
        total_active = n_pos + n_neg  # (B, 1)

        # Avoid division by zero; if no positions, weight will be 0.
        weight = torch.where(total_active > 0, 1.0 / total_active, torch.zeros_like(total_active))

        # Broadcast weight to asset dimension and assign signs
        actions = torch.zeros_like(momentum, dtype=momentum.dtype)
        actions = torch.where(pos_mask, weight.expand_as(momentum), actions)
        actions = torch.where(neg_mask, -weight.expand_as(momentum), actions)

        return actions, torch.zeros_like(actions) 