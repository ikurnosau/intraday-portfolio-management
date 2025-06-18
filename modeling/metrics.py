import torch


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = preds.eq(targets).sum().item()
    return correct / targets.size(0)


def rmse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)

    return torch.sqrt(torch.mean((preds.to(torch.float32) - targets.to(torch.float32)) ** 2))