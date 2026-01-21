import torch
import torch.nn.functional as F

def coral_loss(thresh_logits, targets, reduction='mean'):
    """
    CORAL (Cumulative Ordinal Regression) loss.

    Args:
        thresh_logits: (batch, n_asset, K-1) raw logits for thresholds.
        targets: (batch, n_asset) integer labels in {0, ..., K-1}.
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        Scalar loss (if reduction != 'none') or loss per sample (if 'none').
    """
    device = thresh_logits.device
    K_minus_1 = thresh_logits.size(-1)

    # Binary target for each threshold: 1 if y > t else 0
    thresholds = torch.arange(K_minus_1, device=device).view(1, 1, -1)  # (1, 1, K-1)
    target_bin = (targets.unsqueeze(-1) > thresholds).float()           # (B, A, K-1)

    # BCE with logits across thresholds
    bce = F.binary_cross_entropy_with_logits(thresh_logits, target_bin, reduction='none')  # (B, A, K-1)
    loss_per_sample = bce.sum(dim=-1)  # sum over thresholds → (B, A)

    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:  # 'none'
        return loss_per_sample


class PositionReturnLoss: 
    def __init__(self, fee=0.001): 
        self.fee = fee

    def __call__(self, position: torch.Tensor, next_return: torch.Tensor): 
        return -torch.log1p((position * next_return).sum(dim=-1) - self.fee).mean()

class RiskAdjustedPositionReturnLoss:
    def __init__(self, fee: float = 1e-3, risk_lambda: float = 0.0):
        self.fee = fee
        self.risk_lambda = risk_lambda

    def __call__(self, position: torch.Tensor, next_return: torch.Tensor):
        # portfolio return per sample (simple return)
        port_ret = (position * next_return).sum(dim=-1) - self.fee    # (B,)

        # compute log returns: r = log(1 + p)
        log_ret = torch.log1p(port_ret)                              # (B,)

        # 1. reward term = -E[log(1+p)]
        reward_term = -log_ret.mean()

        # 2. risk term = lambda * Var[log(1+p)]
        if self.risk_lambda > 0:
            demeaned = log_ret - log_ret.mean()
            var = (demeaned ** 2).mean()
            risk_term = self.risk_lambda * var
        else:
            risk_term = 0.0

        return reward_term + risk_term



def position_return_loss_with_entropy(position: torch.Tensor, next_return: torch.Tensor, lambda_entropy: float = 0.003):
    base_loss = -(torch.log1p((position * next_return).sum(dim=-1))).mean()

    p = position.abs() / (position.abs().sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * (p + 1e-12).log()).sum(dim=-1).mean()

    return base_loss + lambda_entropy * entropy