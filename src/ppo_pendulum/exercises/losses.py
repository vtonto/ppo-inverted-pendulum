"""Stage 4: implement the PPO-Clip actor loss and critic loss."""

from __future__ import annotations

import torch


def ppo_clipped_actor_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss_to_minimize, probability_ratio)``.

    TODO(student):
    1. ratio = exp(new_log_prob - old_log_prob)
    2. build unclipped and clipped surrogate objectives
    3. take the elementwise minimum, then negate its batch mean

    ``old_log_prob`` and ``advantages`` must be fixed rollout data.
    """
    raise NotImplementedError("TODO: implement PPO clipped actor loss")


def value_loss(predicted_values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return ``0.5 * mean((predicted_values - targets) ** 2)``."""
    # TODO(student): targets should be detached before or inside this function.
    raise NotImplementedError("TODO: implement critic value loss")

