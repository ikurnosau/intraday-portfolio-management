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
