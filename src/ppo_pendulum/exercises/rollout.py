"""Stage 5: write environment interaction and rollout collection."""

from __future__ import annotations

from typing import Any


def collect_rollout(*, env: Any, actor_critic: Any, observation: Any, steps: int) -> Any:
    """Collect ``steps`` transitions and return a batch plus the latest state.

    TODO(student) checklist:
    - Convert observation to float32 tensor with shape [1, obs_dim].
    - Sample ``raw_action`` and store its joint old log-probability.
    - Clip only the action sent into the Box environment; keep ``raw_action``
      for later log-prob evaluation. This project uses a simple Gaussian and
      clipping approximation, not a tanh-transformed distribution.
    - Save reward, V(s_t), terminated, truncated, and V(s_{t+1}).
    - Evaluate V(s_{t+1}) before resetting a finished environment.
    - Reset on ``terminated or truncated``.
    - Compute GAE only after the time-ordered rollout is complete.

    Start from ``reference/trainer.py:PPOTrainer.collect_rollout`` if stuck.
    """
    raise NotImplementedError("TODO: implement rollout collection")

