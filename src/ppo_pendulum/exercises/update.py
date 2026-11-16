"""Completed PPO mini-batch, multi-epoch update."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ppo_pendulum.exercises.losses import ppo_clipped_actor_loss, value_loss
from ppo_pendulum.exercises.rollout import RolloutBatch


def update_policy(
    *,
    actor_critic: Any,
    rollout_batch: RolloutBatch,
    config: Any,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Update Actor and Critic from fixed rollout data."""
    advantages = rollout_batch.advantages.detach()
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    batch_size = rollout_batch.observations.shape[0]
    device = rollout_batch.observations.device
    metrics: dict[str, list[float]] = {
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(config.update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, config.minibatch_size):
            minibatch = indices[start : start + config.minibatch_size]
            new_log_prob, entropy, predicted_value = actor_critic.evaluate_actions(
                rollout_batch.observations[minibatch],
                rollout_batch.raw_actions[minibatch],
            )
            old_log_prob = rollout_batch.old_log_probs[minibatch].detach()
            actor_loss, ratio = ppo_clipped_actor_loss(
                new_log_prob,
                old_log_prob,
                advantages[minibatch],
                config.clip_coef,
            )
            actor_objective = actor_loss - config.entropy_coef * entropy.mean()
            critic_loss = value_loss(
                predicted_value, rollout_batch.returns[minibatch]
            )

            actor_optimizer.zero_grad(set_to_none=True)
            actor_objective.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor_critic.actor_parameters()), config.max_grad_norm
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor_critic.critic_parameters()), config.max_grad_norm
            )
            critic_optimizer.step()

            with torch.no_grad():
                log_ratio = new_log_prob - old_log_prob
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_coef
                ).float().mean()
            metrics["actor_loss"].append(float(actor_loss.detach()))
            metrics["critic_loss"].append(float(critic_loss.detach()))
            metrics["entropy"].append(float(entropy.mean().detach()))
            metrics["approx_kl"].append(float(approx_kl))
            metrics["clip_fraction"].append(float(clip_fraction))

    return {name: float(np.mean(values)) for name, values in metrics.items()}
