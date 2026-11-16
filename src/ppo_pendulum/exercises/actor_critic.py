"""Completed factorized Gaussian Actor and Critic implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.mean_net(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def joint_log_prob(self, dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        """Return one joint log-density per sample, shape ``[batch]``."""
        return dist.log_prob(actions).sum(dim=-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        return action, self.joint_log_prob(dist, action)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)


class ExerciseActorCritic(nn.Module):
    """Combine the exercise Actor and Critic behind the trainer interface."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.actor = GaussianActor(obs_dim, action_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return self.actor.distribution(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        return raw_action, self.actor.joint_log_prob(dist, raw_action), self.value(obs)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = self.actor.joint_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, self.value(obs)

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.mean_net(obs)

    def actor_parameters(self):
        yield from self.actor.parameters()

    def critic_parameters(self):
        yield from self.critic.parameters()

