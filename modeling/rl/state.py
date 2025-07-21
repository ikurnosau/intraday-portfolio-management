from dataclasses import dataclass
import torch


@dataclass
class State:
    """Container object describing the environment at a single time-step."""

    signal_features: torch.Tensor
    next_step_return: torch.Tensor
    spread: torch.Tensor
    position: torch.Tensor
