import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)


def rmse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)

    return torch.sqrt(torch.mean((preds.to(torch.float32) - targets.to(torch.float32)) ** 2))


# -------------------------------------------------------------------------------------------
# Multi-asset helpers (outputs shape = [batch, asset, class])
# -------------------------------------------------------------------------------------------


def accuracy_multi_asset(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy when model returns per-asset logits.

    Args:
        outputs: (batch, asset, n_class) logits
        targets: (batch, asset) integer labels
    """
    preds = outputs.argmax(dim=-1)  # (batch, asset)
    correct = preds.eq(targets).sum().item()
    return correct / targets.numel()


def rmse_multi_asset(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute RMSE for multi-asset logits.

    Args:
        outputs: (batch, asset, n_class) logits
        targets: (batch, asset) integer labels
    """
    preds = outputs.argmax(dim=-1).to(torch.float32)
    targ = targets.to(torch.float32)
    return torch.sqrt(torch.mean((preds - targ) ** 2))