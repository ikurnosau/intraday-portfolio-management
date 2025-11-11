import torch

from ..modeling_utils import smooth_abs
from .tsa_classifier import TemporalSpatial


class TSAllocator(TemporalSpatial):
    def __init__(self, exploration_epsilon: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_epsilon = exploration_epsilon
        self.training = True

    def forward(self, x: torch.Tensor):
        action_pred = super().forward(x)
        if self.training and self.exploration_epsilon > 0.0:
            noise = self.exploration_epsilon * torch.randn_like(action_pred)
            action_pred = action_pred + noise

        
        action = action_pred / (smooth_abs(action_pred).sum(dim=-1, keepdim=True) + 1e-8)

        return action