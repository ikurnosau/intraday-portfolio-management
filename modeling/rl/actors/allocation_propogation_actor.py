from __future__ import annotations

import torch
import torch.nn as nn
# numpy not required; using torch.topk instead

from ..state import State
from .base_actor import BaseActor


class AllocationPropogationActor(nn.Module, BaseActor):
    """
    Simple actor that uses a signal predictor to trade the most promising assets.
    """

    def __init__(
        self,
        allocator: nn.Module,
        train_allocator: bool = False,
    ):
        super().__init__()
        self.allocator = allocator

        # Freeze allocator parameters (assumed pre-trained)
        self.train_allocator = train_allocator
        if not self.train_allocator:
            for p in self.allocator.parameters():
                p.requires_grad = False
            self.allocator.eval()

    def forward(self, state: State):
        if not self.train_allocator:
            with torch.no_grad():
                action_pred = self.allocator(state.signal_features)  # (B, n_assets)
        else:
            action_pred = self.allocator(state.signal_features)  # (B, n_assets)

        return action_pred, torch.zeros_like(action_pred)

    def train(self, mode: bool = True):
        """Override nn.Module.train to keep the frozen signal predictor in evaluation
        mode while still letting the rest of the actor switch as normal."""
        super().train(mode)
        if not self.train_allocator:
            self.allocator.eval()
        return self
