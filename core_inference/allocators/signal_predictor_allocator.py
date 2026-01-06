from __future__ import annotations

import torch
import torch.nn as nn
# numpy not required; using torch.topk instead


class SignalPredictorAllocator(nn.Module):
    """
    Simple actor that uses a signal predictor to trade the most promising assets.
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        trade_asset_count: int = 1,
        allow_short_positions: bool = True,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.trade_asset_count = trade_asset_count
        self.allow_short_positions = allow_short_positions

    def forward(self, signal_features: torch.Tensor) -> torch.Tensor:
        signal_repr = self.signal_predictor(signal_features)
        # Re-center predictor output: values >0 ⇒ long, <0 ⇒ short
        ls_score = signal_repr - 0.5  # (B, n_assets)

        # Select assets with largest absolute score
        _, top_idx = (ls_score.abs() if self.allow_short_positions else ls_score).topk(self.trade_asset_count, dim=1)  # (B, K)
        mask = torch.zeros_like(ls_score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        selected = torch.where(mask, ls_score, torch.zeros_like(ls_score))

        # Normalise so that Σ|a_i| = 1
        action = selected / (selected.abs().sum(dim=1, keepdim=True) + 1e-8)

        return action