import torch
from typing import List


class SumLogReturnLoss:
    def __init__(self, use_baseline: bool = True):
        self.use_baseline = use_baseline

    def __call__(self, rewards: List[torch.Tensor]) -> torch.Tensor:
        return sum_log_return_loss(rewards, self.use_baseline)

def sum_log_return_loss(
    rewards: List[torch.Tensor],
    use_baseline: bool = True,
) -> torch.Tensor:
    # Shape: (T, batch)
    rewards_t = torch.stack(rewards, dim=0)

    # Sum over the trajectory → (batch,)
    trajectory_log_ret = torch.log1p(rewards_t).sum(dim=0)

    if use_baseline:
        baseline = trajectory_log_ret.mean().detach()
        advantage = trajectory_log_ret - baseline
        loss = -advantage.mean()
    else:
        loss = -trajectory_log_ret.mean()

    return loss


def log_cumulative_trajectory_return_loss(rewards: list[torch.Tensor]) -> torch.Tensor:
    # Shape → (T, batch_size)
    rewards_t = torch.stack(rewards)
    cumulative_return = torch.clamp(1.0 + rewards_t, min=1e-6).prod(dim=0)  # (batch_size,)

    return -torch.log(cumulative_return).mean() 