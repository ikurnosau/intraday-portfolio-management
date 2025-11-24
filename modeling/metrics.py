import torch
import torch.nn.functional as F


def rmse_regression(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Root mean squared error for *regression* outputs.

    Accepts tensors of arbitrary but matching shape. Squeezes a trailing
    singleton dimension (e.g., ``(..., 1)``) in *outputs* if present so the
    shapes align with *targets*.
    """
    if outputs.ndim == targets.ndim + 1 and outputs.shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    return torch.sqrt(torch.mean((outputs.to(torch.float32) - targets.to(torch.float32)) ** 2)).item()


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)


def rmse_classification(outputs: torch.Tensor, targets: torch.Tensor) -> float:
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


def rmse_multi_asset_classification(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute RMSE for multi-asset logits.

    Args:
        outputs: (batch, asset, n_class) logits
        targets: (batch, asset) integer labels
    """
    preds = outputs.argmax(dim=-1).to(torch.float32)
    targ = targets.to(torch.float32)
    return torch.sqrt(torch.mean((preds - targ) ** 2))


def coral_pred_class(outputs: torch.Tensor) -> torch.Tensor:
    # outputs: (B, A, K-1) threshold logits
    probs_ge = torch.sigmoid(outputs)          # P(y > k)
    return (probs_ge > 0.5).sum(dim=-1)        # (B, A) in {0..K-1}

def accuracy_multi_asset_coral(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = coral_pred_class(outputs)
    correct = preds.eq(targets).sum().item()
    return correct / targets.numel()

def rmse_multi_asset_coral(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = coral_pred_class(outputs).to(torch.float32)
    targ = targets.to(torch.float32)
    return torch.sqrt(torch.mean((preds - targ) ** 2)).item()

class MeanReturn: 
    def __init__(self, fee=0.001): 
        self.fee = fee

    def __call__(self, position: torch.Tensor, next_return: torch.Tensor): 
        return ((position * next_return).sum(dim=-1) - self.fee).mean()
