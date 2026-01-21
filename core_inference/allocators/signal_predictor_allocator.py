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
        select_from_n_best: int | None = None,
        confidence_threshold: float = 0.0,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.trade_asset_count = trade_asset_count
        self.allow_short_positions = allow_short_positions
        self.select_from_n_best = select_from_n_best
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        signal_features: torch.Tensor,
        spread: torch.Tensor,
        volatility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal_repr = self.signal_predictor(signal_features)
        # Re-center predictor output: values >0 ⇒ long, <0 ⇒ short
        ls_score = signal_repr - 0.5  # (B, n_assets)
        if not self.allow_short_positions:
            ls_score = ls_score.clamp(min=0) + 1e-5
        _, n_assets = ls_score.shape
        n_best = n_assets if self.select_from_n_best is None else min(self.select_from_n_best, n_assets)
        n_best = max(int(n_best), 1)

        # Narrow investable universe to the assets with the highest "energy_to_friction":
        # volatility / spread (higher is better).
        valid_spread = torch.isfinite(spread) & (spread > 0)
        ratio = volatility / (spread + 1e-8)  # (B, n_assets)
        ratio = ratio.masked_fill(~valid_spread, float("-inf"))
        _, best_idx = torch.topk(ratio, k=n_best, dim=1)  # (B, n_best)
        universe_mask = torch.zeros_like(ls_score, dtype=torch.bool)
        universe_mask.scatter_(1, best_idx, True)
        # If a row has no valid spreads, bypass the universe narrowing entirely for that row.
        valid_any = valid_spread.any(dim=1)  # (B,)
        if (~valid_any).any():
            universe_mask[~valid_any] = True

        # Select assets with largest absolute score, but only from this narrowed universe.
        masked_abs = ls_score.abs().masked_fill(~universe_mask, float("-inf"))

        k = min(self.trade_asset_count, n_best)
        _, top_idx = masked_abs.topk(k, dim=1)  # (B, K)
        mask = torch.zeros_like(ls_score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        selected = torch.where(mask, ls_score, torch.zeros_like(ls_score))

        # Normalise so that Σ|a_i| = 1
        action = selected / (selected.abs().sum(dim=1, keepdim=True) + 1e-8)

        # Confidence per batch element: top absolute ls_score among the (optionally)
        # narrowed universe.
        confidence = masked_abs.max(dim=1).values  # (B,)
        confidence = torch.where(torch.isfinite(confidence), confidence, torch.zeros_like(confidence))

        # If confidence is too low, do not take risk: zero-out the whole action row.
        take_trade = (confidence >= self.confidence_threshold).to(action.dtype).unsqueeze(1)  # (B, 1)
        action = action * take_trade

        return action, confidence