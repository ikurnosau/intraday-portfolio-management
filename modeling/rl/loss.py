import torch


def log_cumulative_trajectory_return_loss(rewards: list[torch.Tensor]) -> torch.Tensor:
    # Shape → (T, batch_size)
    rewards_t = torch.stack(rewards)
    
    # Cumulative multiplicative return per sample: Π_t (1 + r_t)
    cumulative_return = torch.clamp(1.0 + rewards_t, min=1e-6).prod(dim=0)  # (batch_size,)

    # Maximise cumulative_return ⇔ minimise negative log of it
    return -torch.log(cumulative_return).mean() 