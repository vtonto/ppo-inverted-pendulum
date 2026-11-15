"""Stage 6: write PPO mini-batch, multi-epoch updates."""

from __future__ import annotations

from typing import Any


def update_policy(*, actor_critic: Any, rollout_batch: Any, config: Any) -> dict[str, float]:
    """Update Actor and Critic using fixed rollout data.

    TODO(student) checklist:
    - Normalize advantages across the rollout (optional but recommended).
    - Shuffle indices each epoch; do not reorder time before GAE is computed.
    - Re-evaluate the stored actions under the current policy.
    - Use stored old log-probs in the ratio; never recompute the old policy.
    - Optimize the clipped Actor objective and value regression objective.
    - Keep advantages and critic targets detached.
    - Clip gradient norm and report actor loss, critic loss, KL estimate, and
      clip fraction.
    """
    raise NotImplementedError("TODO: implement PPO update")

