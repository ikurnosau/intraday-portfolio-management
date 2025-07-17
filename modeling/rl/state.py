from dataclasses import dataclass
import torch


@dataclass
class State:
    """Container object describing the environment at a single time-step."""

    signal_features: torch.Tensor
    next_step_return: torch.Tensor
    spread: torch.Tensor
    position: torch.Tensor

    def copy(self) -> "State":
        """Return a deep copy of the state instance."""
        return State(
            signal_features=self.signal_features.detach().clone(),
            next_step_return=self.next_step_return.detach().clone(),
            spread=self.spread.detach().clone(),
            position=self.position.clone(),  # clone to avoid mutating template while retaining grad
        )
