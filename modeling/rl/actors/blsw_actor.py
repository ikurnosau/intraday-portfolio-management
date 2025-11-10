from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class BLSWActor(nn.Module, BaseActor):
    """
    Buying-Loser-Selling-Winner (cross-sectional mean-reversion) actor.

    Over a *look_back_window* it computes each asset's cumulative log return.
    It then goes **long** the `trade_asset_count` worst performers ("losers")
    and **short** the same number of best performers ("winners").  All
    selected positions are equally weighted so the allocation vector *a*
    follows

        Σ |aᵢ| = 1, aᵢ ∈ [-1, 1].
    """

    def __init__(self, look_back_window: int = 20, trade_asset_count: int = 5):
        super().__init__()
        self.look_back_window = look_back_window
        self.trade_asset_count = trade_asset_count

    def forward(self, state: State):
        # signal_features: (B, n_assets, seq_len, n_features)
        log_returns = state.signal_features[..., 0][..., -self.look_back_window:]  # (B, n_assets, L)
        cum_log_ret = log_returns.sum(dim=-1)  # (B, n_assets)

        batch_size, n_assets = cum_log_ret.shape
        k = min(self.trade_asset_count, n_assets // 2)
        actions = torch.zeros_like(cum_log_ret)
        if k == 0:
            return actions

        weight = 1.0 / (2 * k)

        # Losers: smallest cumulative return → go long
        loser_idx = cum_log_ret.topk(k, dim=1, largest=False).indices  # (B, k)
        # Winners: largest cumulative return → go short
        winner_idx = cum_log_ret.topk(k, dim=1, largest=True).indices  # (B, k)

        actions.scatter_(1, loser_idx, weight)
        actions.scatter_(1, winner_idx, -weight)

        return actions, torch.zeros_like(actions)