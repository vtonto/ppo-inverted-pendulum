"""Completed GAE with correct episode-boundary semantics."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached advantages and Critic targets, both shape ``[T]``."""
    tensors = (rewards, values, next_values, terminated, truncated)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("all GAE inputs must have shape [T]")
    if len({tensor.shape for tensor in tensors}) != 1:
        raise ValueError("all GAE inputs must have the same shape")

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(len(rewards))):
        bootstrap_mask = 1.0 - terminated[t].to(rewards.dtype)
        recursion_mask = 1.0 - torch.logical_or(
            terminated[t], truncated[t]
        ).to(rewards.dtype)
        delta = rewards[t] + gamma * bootstrap_mask * next_values[t] - values[t]
        last_gae = delta + gamma * gae_lambda * recursion_mask * last_gae
        advantages[t] = last_gae

    targets = advantages + values
    return advantages.detach(), targets.detach()

