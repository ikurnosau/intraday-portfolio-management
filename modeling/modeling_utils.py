import torch


def smooth_abs(x: torch.Tensor, eps_ratio: float = 1e-3) -> torch.Tensor:
    eps = eps_ratio * torch.max(x.abs().max(), torch.tensor(1e-12))
    abs_val =  torch.sqrt(x.pow(2) + eps**2) - eps
    return abs_val