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
        train_signal_predictor: bool = False,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.trade_asset_count = trade_asset_count

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

        # Select assets with largest absolute score
        _, top_idx = ls_score.abs().topk(self.trade_asset_count, dim=1)  # (B, K)
        mask = torch.zeros_like(ls_score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        selected = torch.where(mask, ls_score, torch.zeros_like(ls_score))

        # Normalise so that Σ|a_i| = 1
        action = selected / (selected.abs().sum(dim=1, keepdim=True) + 1e-8)

        return action

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)
        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self
