import torch
import torch.nn.functional as F

from ..modeling_utils import smooth_abs
from .tsa_classifier import TemporalSpatial


class TSAllocator(TemporalSpatial):
    # def __init__(self, temperature = 0.3, *args, **kwargs):
    def __init__(self, temperature = 1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        
    def forward(self, x: torch.Tensor):
        action_pred = super().forward(x)
        action_pred = F.sigmoid(action_pred / self.temperature)
  
        action = action_pred / action_pred.sum(dim=-1, keepdim=True)

        return action