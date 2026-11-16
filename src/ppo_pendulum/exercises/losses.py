"""Completed PPO-Clip Actor loss and Critic value loss."""

from __future__ import annotations

import torch


def ppo_clipped_actor_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss_to_minimize, probability_ratio)``."""
    fixed_old_log_prob = old_log_prob.detach()
    fixed_advantages = advantages.detach()
    ratio = torch.exp(new_log_prob - fixed_old_log_prob)
    unclipped = ratio * fixed_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * fixed_advantages
    return -torch.minimum(unclipped, clipped).mean(), ratio


def value_loss(predicted_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return ``0.5 * mean((predicted_values - targets) ** 2)``."""
    return 0.5 * torch.mean((predicted_values - targets.detach()) ** 2)

