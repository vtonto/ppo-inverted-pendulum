"""Stage 3: implement GAE with correct episode-boundary semantics."""

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
    """Return ``(advantages, critic_targets)``, both shape ``[T]``.

    TODO(student): traverse from T-1 to 0.

    Hints:
    - Bootstrap mask for delta: ``1 - terminated[t]``.
    - Recursion mask: ``1 - (terminated[t] or truncated[t])``.
    - ``critic_targets = advantages + values``.
    - Stored rollout data and targets should not retain an autograd graph.
    """
    raise NotImplementedError("TODO: implement GAE")

