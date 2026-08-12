import torch
import torch.nn as nn
from collections import OrderedDict


def smooth_abs(x: torch.Tensor, eps_ratio: float = 1e-3) -> torch.Tensor:
    eps = eps_ratio * torch.max(x.abs().max(), torch.tensor(1e-12))
    abs_val =  torch.sqrt(x.pow(2) + eps**2) - eps
    return abs_val


def print_model_parameters(model: nn.Module):
    module_params = OrderedDict()

    for module_name, module in model.named_modules():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if count > 0:
            module_params[module_name or "[ROOT]"] = count

    print(f"{'Module':<40} Params")
    print("-" * 60)
    for name, count in module_params.items():
        print(f"{name:<40} {count}")