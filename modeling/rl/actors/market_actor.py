from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class MarketActor(nn.Module, BaseActor):
    """
    Market actor.

    For each asset we compute the cumulative log return over the given
    *look_back_window*.  We go **long** if the return is positive and
    **short** if the return is negative.  All selected positions are
    equally-weighted such that the allocation vector *a* satisfies

    Assets with zero return receive zero allocation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state: State):

        # Broadcast weight to asset dimension and assign signs
        actions = torch.ones_like(state.position, dtype=state.position.dtype)
        actions = actions / state.position.shape[-1]

        return actions , torch.zeros_like(actions)