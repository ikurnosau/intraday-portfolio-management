from abc import ABC, abstractmethod
import torch

from modeling.modeling_utils import smooth_abs

from .state import State


class BaseReward(ABC):
    def __init__(self, fee: float, spread_multiplier: float = 0.33):
        self.fee = fee
        self.spread_multiplier = spread_multiplier

    @abstractmethod
    def __call__(self, current_state: State, next_state: State) -> torch.Tensor:
        pass


class EstimatedReturnReward(BaseReward):
    def __call__(self, current_state: State, next_state: State) -> torch.Tensor:
        return_component = next_state.position * current_state.next_step_return  # (batch_size, assets)
        cost_component = smooth_abs(next_state.position - current_state.position) * (self.fee + (current_state.spread / 2) * self.spread_multiplier)  # (batch_size, assets)
        return (return_component - cost_component).sum(dim=-1)  # (batch_size,)
