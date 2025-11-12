import torch
import torch.nn.functional as F

from ..modeling_utils import smooth_abs
from .tsa_classifier import TemporalSpatial


class TSAllocator(TemporalSpatial):
    # def __init__(self, temperature = 0.3, *args, **kwargs):
    def __init__(self, temperature = 1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor, log=False):
        action_pred = super().forward(x)
        action_pred = F.softsign(action_pred / self.temperature)
  
        action = action_pred / (smooth_abs(action_pred).sum(dim=-1, keepdim=True) + 1e-15)

        total_allocations = x.abs().sum(dim=-1)
        assert (total_allocations < 1.1).all(), f"Some allocations are greater than 1.1 in total! Allocations: {total_allocations}"

        return action