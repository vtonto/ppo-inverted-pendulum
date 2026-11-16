from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo_pendulum.config import PPOConfig
from ppo_pendulum.reference.algorithm import (
    compute_gae,
    ppo_clipped_actor_loss,
    value_loss,
)
from ppo_pendulum.reference.model import GaussianActorCritic


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    raw_actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PPOTrainer:
    implementation = "reference"

    def __init__(self, config: PPOConfig) -> None:
        config.validate()
        self.config = config
        seed_everything(config.seed)
        self.device = torch.device(config.device)
        self.env = gym.make(config.env_id, max_episode_steps=config.max_episode_steps)
        self.env.action_space.seed(config.seed)
        self.obs, _ = self.env.reset(seed=config.seed)

        if len(self.env.observation_space.shape) != 1 or len(self.env.action_space.shape) != 1:
            raise ValueError("this teaching implementation expects flat Box spaces")
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        self.model = GaussianActorCritic(obs_dim, action_dim, config.hidden_size).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=config.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=config.critic_lr, eps=1e-5
        )
        self.action_low = torch.as_tensor(
            self.env.action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            self.env.action_space.high, dtype=torch.float32, device=self.device
        )
        self.global_step = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []

    def collect_rollout(self, steps: int) -> RolloutBatch:
        observations: list[torch.Tensor] = []
        raw_actions: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        values: list[torch.Tensor] = []
        next_values: list[torch.Tensor] = []
        terminated_flags: list[bool] = []
        truncated_flags: list[bool] = []

        for _ in range(steps):
            obs_tensor = torch.as_tensor(
                self.obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                raw_action, log_prob, value = self.model.act(obs_tensor)
            env_action_tensor = torch.clamp(raw_action[0], self.action_low, self.action_high)
            next_obs, reward, terminated, truncated, _ = self.env.step(
                env_action_tensor.cpu().numpy()
            )
            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(
                    next_obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                next_value = self.model.value(next_obs_tensor)[0]

            observations.append(obs_tensor[0])
            raw_actions.append(raw_action[0])
            old_log_probs.append(log_prob[0])
            rewards.append(float(reward))
            values.append(value[0])
            next_values.append(next_value)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))

            self.global_step += 1
            self.episode_return += float(reward)
            self.episode_length += 1
            done = terminated or truncated
            if done:
                self.completed_returns.append(self.episode_return)
                self.completed_lengths.append(self.episode_length)
                self.episode_return = 0.0
                self.episode_length = 0
                self.obs, _ = self.env.reset()
            else:
                self.obs = next_obs

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.stack(values)
        next_values_t = torch.stack(next_values)
        terminated_t = torch.tensor(terminated_flags, dtype=torch.bool, device=self.device)
        truncated_t = torch.tensor(truncated_flags, dtype=torch.bool, device=self.device)
        advantages, returns = compute_gae(
            rewards_t,
            values_t,
            next_values_t,
            terminated_t,
            truncated_t,
            self.config.gamma,
            self.config.gae_lambda,
        )
        return RolloutBatch(
            observations=torch.stack(observations),
            raw_actions=torch.stack(raw_actions),
            old_log_probs=torch.stack(old_log_probs),
            advantages=advantages,
            returns=returns,
            values=values_t,
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        cfg = self.config
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        batch_size = batch.observations.shape[0]
        metrics: dict[str, list[float]] = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for _ in range(cfg.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = indices[start : start + cfg.minibatch_size]
                new_log_prob, entropy, predicted_value = self.model.evaluate_actions(
                    batch.observations[mb], batch.raw_actions[mb]
                )
                actor_loss, ratio, clip_fraction = ppo_clipped_actor_loss(
                    new_log_prob,
                    batch.old_log_probs[mb],
                    advantages[mb],
                    cfg.clip_coef,
                )
                actor_objective = actor_loss - cfg.entropy_coef * entropy.mean()
                critic_loss = value_loss(predicted_value, batch.returns[mb])

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.actor_parameters()), cfg.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.critic_parameters()), cfg.max_grad_norm
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - batch.old_log_probs[mb]
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                metrics["actor_loss"].append(float(actor_loss.detach()))
                metrics["critic_loss"].append(float(critic_loss.detach()))
                metrics["entropy"].append(float(entropy.mean().detach()))
                metrics["approx_kl"].append(float(approx_kl))
                metrics["clip_fraction"].append(float(clip_fraction))

        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.as_serializable_dict(),
                "global_step": self.global_step,
                "obs_dim": int(np.prod(self.env.observation_space.shape)),
                "action_dim": int(np.prod(self.env.action_space.shape)),
                "implementation": self.implementation,
            },
            path,
        )

    def close(self) -> None:
        self.env.close()


def train(
    config: PPOConfig, trainer_class: type[PPOTrainer] = PPOTrainer
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(config.as_serializable_dict(), indent=2), encoding="utf-8"
    )
    trainer = trainer_class(config)
    metrics_path = config.output_dir / "metrics.csv"
    start_time = time.perf_counter()
    recent_episode_index = 0

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "step",
                    "actor_loss",
                    "critic_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                    "mean_recent_return",
                    "mean_recent_length",
                    "steps_per_second",
                ],
            )
            writer.writeheader()
            while trainer.global_step < config.total_timesteps:
                steps = min(
                    config.rollout_steps, config.total_timesteps - trainer.global_step
                )
                batch = trainer.collect_rollout(steps)
                stats = trainer.update(batch)
                new_returns = trainer.completed_returns[recent_episode_index:]
                new_lengths = trainer.completed_lengths[recent_episode_index:]
                recent_episode_index = len(trainer.completed_returns)
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                row = {
                    "step": trainer.global_step,
                    **stats,
                    "mean_recent_return": float(np.mean(new_returns)) if new_returns else "",
                    "mean_recent_length": float(np.mean(new_lengths)) if new_lengths else "",
                    "steps_per_second": trainer.global_step / elapsed,
                }
                writer.writerow(row)
                file.flush()
                print(
                    f"step={trainer.global_step:>7d} "
                    f"actor={stats['actor_loss']:+.4f} "
                    f"critic={stats['critic_loss']:.4f} "
                    f"episodes={len(trainer.completed_returns)}"
                )
        checkpoint = config.output_dir / "final.pt"
        trainer.save(checkpoint)
        return checkpoint
    finally:
        trainer.close()

