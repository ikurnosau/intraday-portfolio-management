from abc import ABC, abstractmethod

import torch

from ..state import State


class BaseActor(ABC):
    @abstractmethod
    def forward(self, state: State) -> tuple[torch.Tensor, torch.Tensor]:
        pass