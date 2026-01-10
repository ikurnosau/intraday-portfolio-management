from __future__ import annotations

import torch
import torch.nn as nn
# numpy not required; using torch.topk instead

from ..state import State
from .base_actor import BaseActor


class SignalPredictorActor(nn.Module, BaseActor):
    """
    Simple actor that uses a signal predictor to trade the most promising assets.
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        trade_asset_count: int = 5,
        select_from_n_best: int | None = None,
        train_signal_predictor: bool = False,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.trade_asset_count = trade_asset_count
        self.select_from_n_best = select_from_n_best

        # Freeze predictor parameters (assumed pre-trained)
        self.train_signal_predictor = train_signal_predictor
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

    def forward(self, state: State):
        if not self.train_signal_predictor:
            with torch.no_grad():
                signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)
        else:
            signal_repr = self.signal_predictor(state.signal_features)  # (B, n_assets)
        # Re-center predictor output: values >0 ⇒ long, <0 ⇒ short
        ls_score = signal_repr - 0.5  # (B, n_assets)

        batch_size, n_assets = ls_score.shape

        # Narrow investable universe to the assets with the highest "energy_to_friction":
        # volatility / spread (higher is better).
        ratio = state.volatility / (state.spread + 1e-8)  # (B, n_assets)
        n_best = n_assets if self.select_from_n_best is None else min(self.select_from_n_best, n_assets)
        n_best = max(int(n_best), 1)

        _, best_idx = torch.topk(ratio, k=n_best, dim=1)  # (B, n_best)
        universe_mask = torch.zeros_like(ls_score, dtype=torch.bool)
        universe_mask.scatter_(1, best_idx, True)

        # Select assets with largest absolute score, but only from this narrowed universe
        k = min(self.trade_asset_count, n_best)
        masked_abs = ls_score.abs().masked_fill(~universe_mask, float("-inf"))
        _, top_idx = masked_abs.topk(k, dim=1)  # (B, K)
        mask = torch.zeros_like(ls_score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        selected = torch.where(mask, ls_score, torch.zeros_like(ls_score))

        # Normalise so that Σ|a_i| = 1
        action = selected / (selected.abs().sum(dim=1, keepdim=True) + 1e-8)

        return action, torch.zeros_like(action)

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)
        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self
