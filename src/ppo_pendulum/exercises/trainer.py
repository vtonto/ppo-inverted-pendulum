"""End-to-end trainer assembled from the completed exercise modules."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo_pendulum.config import PPOConfig
from ppo_pendulum.exercises.actor_critic import ExerciseActorCritic
from ppo_pendulum.exercises.rollout import RolloutBatch, collect_rollout
from ppo_pendulum.exercises.update import update_policy
from ppo_pendulum.reference.trainer import seed_everything, train as run_training_loop


class ExercisePPOTrainer:
    implementation = "exercise"

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
        self.model = ExerciseActorCritic(obs_dim, action_dim, config.hidden_size).to(
            self.device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_parameters(), lr=config.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(), lr=config.critic_lr, eps=1e-5
        )
        self.global_step = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []

    def collect_rollout(self, steps: int) -> RolloutBatch:
        result = collect_rollout(
            env=self.env,
            actor_critic=self.model,
            observation=self.obs,
            steps=steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            episode_return=self.episode_return,
            episode_length=self.episode_length,
        )
        self.obs = result.observation
        self.episode_return = result.episode_return
        self.episode_length = result.episode_length
        self.completed_returns.extend(result.completed_returns)
        self.completed_lengths.extend(result.completed_lengths)
        self.global_step += steps
        return result.batch

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        return update_policy(
            actor_critic=self.model,
            rollout_batch=batch,
            config=self.config,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
        )

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


def train(config: PPOConfig) -> Path:
    return run_training_loop(config, trainer_class=ExercisePPOTrainer)
