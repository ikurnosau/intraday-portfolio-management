from __future__ import annotations

import torch
import torch.nn as nn


class DeterministicOneStockAllocator(nn.Module):
    """Cycles through assets: each forward fully longs stock ``counter % n_assets``."""

    def __init__(self, n_assets: int):
        super().__init__()
        if n_assets < 1:
            raise ValueError("n_assets must be at least 1")
        self.n_assets = n_assets
        self.counter = 0
        self._device_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def reset_counter(self) -> None:
        self.counter = 0

    def forward(
        self,
        x: torch.Tensor,
        spread: torch.Tensor,
        volatility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        action = torch.zeros((batch_size, self.n_assets), device=x.device)
        for batch_i in range(batch_size):
            asset_i = self.counter % self.n_assets
            action[batch_i, asset_i] = 1.0
            self.counter += 1
        confidence = torch.ones(batch_size, device=x.device)
        return action, confidence
