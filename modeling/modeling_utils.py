import torch


def smooth_abs(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x.pow(2) + eps**2) - eps