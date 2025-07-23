
import torch

from .state import State


def estimated_return_reward(current_state: State, next_state: State, fee: float, spread_multiplier: float = 1) -> torch.Tensor:
    return_component = next_state.position * current_state.next_step_return  # (batch_size, assets)
    cost_component = torch.abs(next_state.position - current_state.position) * (fee + (current_state.spread / 2) * spread_multiplier)  # (batch_size, assets)
    return (return_component - cost_component).sum(dim=-1)  # (batch_size,)