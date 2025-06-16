import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)