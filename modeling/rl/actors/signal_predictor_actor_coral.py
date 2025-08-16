from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor


class SignalPredictorActorCoral(nn.Module, BaseActor):
    """
    Actor that uses a CORAL-trained signal predictor producing threshold logits
    with shape (batch, num_assets, K-1). It converts these logits into a
    single scalar signal per asset in [0, 1] via the expected ordinal class,
    then applies the same selection and normalisation as SignalPredictorActor.
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        trade_asset_count: int = 5,
        train_signal_predictor: bool = False,
    ):
        super().__init__()
        self.signal_predictor = signal_predictor
        self.trade_asset_count = trade_asset_count

        self.train_signal_predictor = train_signal_predictor
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

    @staticmethod
    def _coral_expected_normalized(threshold_logits: torch.Tensor) -> torch.Tensor:
        """
        Convert CORAL threshold logits (B, A, K-1) to a scalar signal in [0, 1]
        using the expected class index divided by (K-1):

            E[y] = sum_{k=0}^{K-2} P(y > k) where P(y > k) = sigmoid(logit_k)

        Returns: (B, A) tensor in [0, 1].
        """
        probs_ge = torch.sigmoid(threshold_logits)
        expected_class = probs_ge.sum(dim=-1)
        denom = threshold_logits.size(-1)
        return expected_class / max(denom, 1)

    def forward(self, state: State):
        if not self.train_signal_predictor:
            with torch.no_grad():
                thresh_logits = self.signal_predictor(state.signal_features)  # (B, A, K-1)
        else:
            thresh_logits = self.signal_predictor(state.signal_features)

        signal = self._coral_expected_normalized(thresh_logits)  # (B, A) in [0,1]
        ls_score = signal - 0.5

        _, top_idx = ls_score.abs().topk(self.trade_asset_count, dim=1)
        mask = torch.zeros_like(ls_score, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)

        selected = torch.where(mask, ls_score, torch.zeros_like(ls_score))
        action = selected / (selected.abs().sum(dim=1, keepdim=True) + 1e-8)

        return action

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self

