from __future__ import annotations

import torch
import torch.nn as nn

from ..state import State
from .base_actor import BaseActor
from ...modeling_utils import smooth_abs


class RlActor(nn.Module, BaseActor):
    """Probabilistic policy mapping :class:`State` → (action, log_prob).

    The action *a* satisfies −1 ≤ aᵢ ≤ 1 and ∑|aᵢ| = 1.  Exploration is
    achieved by sampling a sign vector from a Bernoulli distribution and
    a magnitude vector from a Dirichlet distribution on the simplex.
    """

    def __init__(
        self,
        signal_predictor: nn.Module,
        n_assets: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        train_signal_predictor: bool = False,
    ):
        super().__init__()

        self.signal_predictor = signal_predictor
        self.n_assets = n_assets
        self.train_signal_predictor = train_signal_predictor

        # ---- Freeze / unfreeze predictor parameters ----
        if not self.train_signal_predictor:
            for p in self.signal_predictor.parameters():
                p.requires_grad = False
            self.signal_predictor.eval()

        # ---- Shared fully-connected backbone ----
        layers: list[nn.Module] = []
        in_features = n_assets * 3  # predictor output + position + vol/spread
        for i in range(num_layers):
            layers.append(nn.Linear(in_features if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.fc_shared = nn.Sequential(*layers)

        # ---- Policy heads ----
        self.sign_head = nn.Linear(hidden_dim, n_assets)    # Bernoulli logits
        self.weight_head = nn.Linear(hidden_dim, n_assets)  # Dirichlet (α)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, state: State, *, deterministic: bool = False):
        # --- Encode predictor features ---
        if self.train_signal_predictor:
            signal_repr = self.signal_predictor(state.signal_features)
        else:
            with torch.no_grad():
                signal_repr = self.signal_predictor(state.signal_features)

        # --- Compose and normalise feature vector ---
        features = torch.cat(
            [signal_repr, state.position, state.spread], dim=-1
        )

        h = self.fc_shared(features)  # (B, hidden_dim)

        logits_sign = self.sign_head(h)            # (B, n_assets)
        alpha_raw = self.weight_head(h)            # (B, n_assets)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1e-3  # αᵢ > 0
        
        bern_dist = torch.distributions.Bernoulli(logits=logits_sign)
        dir_dist = torch.distributions.Dirichlet(alpha)

        if self.training and not deterministic:
            sign_sample = bern_dist.sample()                 # (B, n_assets) in {0,1}
            sign = 2.0 * sign_sample - 1.0                   # → {-1, +1}
            weights = dir_dist.sample()                      # simplex, positive, sums to 1
        else:
            # Deterministic: use mode / mean estimates
            sign = torch.where(torch.sigmoid(logits_sign) >= 0.5, 1.0, -1.0)
            weights = alpha / alpha.sum(dim=-1, keepdim=True)

        action = sign * weights                              # (B, n_assets)

        # Normalise to ensure Σ|aᵢ| = 1 exactly (numerical safety)
        action = action / (smooth_abs(action).sum(dim=-1, keepdim=True) + 1e-8)

        # Log-probability of the joint sample
        log_prob_sign = bern_dist.log_prob((sign + 1) / 2).sum(-1)  # (B,)
        log_prob_weight = dir_dist.log_prob(weights)                # (B,)
        log_prob = log_prob_sign + log_prob_weight                  # (B,)

        return action, log_prob

    # ------------------------------------------------------------------
    # Keep predictor frozen in .train(False) mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_signal_predictor:
            self.signal_predictor.eval()
        return self
