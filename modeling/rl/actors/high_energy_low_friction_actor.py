from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class HighEnergyLowFrictionActor(nn.Module, BaseActor):
    """
    Actor that trades a single asset per step, keeping the direction suggested by the predictor.

    - Uses a signal predictor to rank assets by confidence (|signal - 0.5|)
    - Takes the top-K assets by confidence
    - Among those, selects the asset with the highest volatility-to-spread ratio
    - Allocates the entire position to that single asset, direction long if signal>0.5 else short
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        top_k: int = 5,
        train_signal_predictor: bool = False,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.top_k = top_k

        # Freeze predictor parameters (assumed pre-trained)
        self.train_signal_predictor = train_signal_predictor
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

    def forward(self, state: State):
        if not self.train_signal_predictor:
            with torch.no_grad():
                signal_scores = self.signal_predictor(state.signal_features)  # (B, n_assets)
        else:
            signal_scores = self.signal_predictor(state.signal_features)  # (B, n_assets)

        batch_size, n_assets = signal_scores.shape

        # Center around 0 to capture long/short direction relative to 0.5
        centered = signal_scores - 0.5  # (B, n_assets)

        # Pick top-K by confidence |signal-0.5|
        k = min(self.top_k, n_assets)
        _, top_idx = torch.topk(centered.abs(), k=k, dim=1)  # (B, K)

        mask = torch.zeros_like(centered, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        # Compute energy/friction ratio and mask to top-K
        ratio = state.volatility / (state.spread + 1e-8)  # (B, n_assets)
        masked_ratio = ratio.masked_fill(~mask, float("-inf"))

        # Choose the single best asset among top-K
        chosen_idx = masked_ratio.argmax(dim=1)  # (B,)

        # Determine direction from predictor for the chosen asset
        direction = torch.where(centered > 0, 1.0, -1.0)  # (B, n_assets)
        chosen_sign = direction.gather(1, chosen_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Build action: full allocation to chosen asset, signed by predictor direction
        action = torch.zeros_like(signal_scores)
        batch_arange = torch.arange(batch_size, device=action.device)
        action[batch_arange, chosen_idx] = chosen_sign

        return action

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)
        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self

