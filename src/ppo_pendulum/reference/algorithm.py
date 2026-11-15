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
    """Compute GAE and detached critic targets for one time-ordered rollout.

    ``terminated`` removes the bootstrap value because the MDP truly ended.
    ``truncated`` keeps the bootstrap in delta, but stops GAE recursion so that
    the next reset episode cannot leak into the current episode.
    """
    tensors = (rewards, values, next_values, terminated, truncated)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("all GAE inputs must be rank-1 tensors [T]")
    if len({t.shape for t in tensors}) != 1:
        raise ValueError("all GAE inputs must have the same shape")

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        bootstrap_mask = 1.0 - terminated[t].to(rewards.dtype)
        episode_continues = 1.0 - torch.logical_or(terminated[t], truncated[t]).to(
            rewards.dtype
        )
        delta = rewards[t] + gamma * bootstrap_mask * next_values[t] - values[t]
        last_gae = delta + gamma * gae_lambda * episode_continues * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages.detach(), returns.detach()


def ppo_clipped_actor_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return minimized actor loss, ratio, and clip fraction."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
    return loss, ratio, clip_fraction


def value_loss(predicted_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Half mean-squared error, matching 1/2 (V-y)^2 notation."""
    return 0.5 * torch.mean((predicted_values - targets) ** 2)

