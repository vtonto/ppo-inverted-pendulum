"""Completed environment interaction and rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ppo_pendulum.exercises.gae import compute_gae


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


@dataclass(slots=True)
class RolloutResult:
    batch: RolloutBatch
    observation: np.ndarray
    episode_return: float
    episode_length: int
    completed_returns: list[float]
    completed_lengths: list[int]


def collect_rollout(
    *,
    env: Any,
    actor_critic: Any,
    observation: np.ndarray,
    steps: int,
    gamma: float,
    gae_lambda: float,
    episode_return: float = 0.0,
    episode_length: int = 0,
) -> RolloutResult:
    """Collect a time-ordered rollout and compute GAE before shuffling.

    The raw Gaussian action is stored for PPO probability evaluation. Only the
    action sent to the bounded environment is clipped.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    device = next(actor_critic.parameters()).device
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(
        env.action_space.high, dtype=torch.float32, device=device
    )
    observations: list[torch.Tensor] = []
    raw_actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    values: list[torch.Tensor] = []
    next_values: list[torch.Tensor] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    current_observation = observation

    for _ in range(steps):
        obs_tensor = torch.as_tensor(
            current_observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            raw_action, log_prob, value = actor_critic.act(obs_tensor)
        env_action = torch.clamp(raw_action[0], action_low, action_high)
        next_observation, reward, terminated, truncated, _ = env.step(
            env_action.cpu().numpy()
        )

        # Compute V(s_{t+1}) before reset. A truncation still bootstraps from it.
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(
                next_observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            next_value = actor_critic.value(next_obs_tensor)[0]

        observations.append(obs_tensor[0])
        raw_actions.append(raw_action[0])
        old_log_probs.append(log_prob[0])
        rewards.append(float(reward))
        values.append(value[0])
        next_values.append(next_value)
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        episode_return += float(reward)
        episode_length += 1
        if terminated or truncated:
            completed_returns.append(episode_return)
            completed_lengths.append(episode_length)
            episode_return = 0.0
            episode_length = 0
            current_observation, _ = env.reset()
        else:
            current_observation = next_observation

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t = torch.stack(values)
    next_values_t = torch.stack(next_values)
    terminated_t = torch.tensor(terminated_flags, dtype=torch.bool, device=device)
    truncated_t = torch.tensor(truncated_flags, dtype=torch.bool, device=device)
    advantages, returns = compute_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminated_t,
        truncated_t,
        gamma,
        gae_lambda,
    )
    batch = RolloutBatch(
        observations=torch.stack(observations).detach(),
        raw_actions=torch.stack(raw_actions).detach(),
        old_log_probs=torch.stack(old_log_probs).detach(),
        advantages=advantages,
        returns=returns,
        values=values_t.detach(),
    )
    return RolloutResult(
        batch=batch,
        observation=current_observation,
        episode_return=episode_return,
        episode_length=episode_length,
        completed_returns=completed_returns,
        completed_lengths=completed_lengths,
    )
