from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class XSMomActor(nn.Module, BaseActor):
    """
    Cross-sectional momentum actor.
    """

    def __init__(
        self,
        look_back_window: int = 20,
        trade_asset_count: int = 5,
    ):
        super().__init__()
        self.trade_asset_count = trade_asset_count
        self.look_back_window = look_back_window

    def forward(self, state: State):
        # Expecting `signal_features` of shape (B, n_assets, seq_len, n_features)
        # Log-returns are stored in feature index 0.
        log_returns = state.signal_features[..., 0][..., -self.look_back_window:]  # (B, n_assets, look_back_window)

        # Momentum score: cumulative log return over the look-back window
        momentum_score = log_returns.sum(dim=-1)  # (B, n_assets)

        batch_size, n_assets = momentum_score.shape
        k = min(self.trade_asset_count, n_assets // 2)  # number of longs/shorts per batch element

        # Allocate tensor for actions
        actions = torch.zeros_like(momentum_score, dtype=momentum_score.dtype)

        if k == 0:
            return actions  # edge-case: not enough assets to trade

        # Equal weight so that Σ|a_i| = 1  → each of 2k positions gets 1/(2k) magnitude
        weight = 1.0 / (2 * k)

        # Long top-k assets, short bottom-k assets for each sample in the batch
        topk_idx = momentum_score.topk(k, dim=1, largest=True).indices  # (B, k)
        bottomk_idx = momentum_score.topk(k, dim=1, largest=False).indices  # (B, k)

        # Scatter the weights
        actions.scatter_(1, topk_idx, weight)
        actions.scatter_(1, bottomk_idx, -weight)

        return actions



