import torch

from ..modeling_utils import smooth_abs
from .tsa_classifier import TemporalSpatial


class TSAllocator(TemporalSpatial):
    def forward(self, x: torch.Tensor):
        action_pred = super().forward(x)
        action = action_pred / (smooth_abs(action_pred).sum(dim=-1, keepdim=True) + 1e-8)

        return action