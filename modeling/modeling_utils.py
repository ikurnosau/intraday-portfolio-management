import torch
import torch.nn as nn
from collections import OrderedDict
from config.experiment_config import ExperimentConfig


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


def load_model_and_allocator_params(model_path: str, device: torch.device, config: ExperimentConfig) -> tuple[torch.nn.Module, dict]:
    """."""

    ckpt = torch.load(model_path, map_location=device)
    
    state_dict = ckpt["model_state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k.replace("_orig_mod.", "")] = v
        else:
            new_state_dict[k] = v
    
    model = config.model_config.model
    model.load_state_dict(new_state_dict)
    allocator_params = ckpt["allocator_params"]

    return model, allocator_params