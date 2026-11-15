from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def _orthogonal_init(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(input_dim: int, hidden_size: int, output_dim: int, output_gain: float) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_dim),
    )
    for layer in net[:-1]:
        _orthogonal_init(layer)
    _orthogonal_init(net[-1], output_gain)
    return net


class GaussianActorCritic(nn.Module):
    """Factorized Gaussian actor and a scalar state-value critic.

    Observations have shape ``[batch, obs_dim]``. Actions have shape
    ``[batch, action_dim]``. ``log_prob`` and ``value`` both have shape
    ``[batch]``.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.actor_mean = _mlp(obs_dim, hidden_size, action_dim, output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = _mlp(obs_dim, hidden_size, 1, output_gain=1.0)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def joint_log_prob(dist: Normal, actions: torch.Tensor) -> torch.Tensor:
        """Return log pi(a|s) by summing per-dimension log densities."""
        return dist.log_prob(actions).sum(dim=-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = self.joint_log_prob(dist, raw_action)
        value = self.value(obs)
        return raw_action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate rollout actions under the current parameters.

        Do not resample here: PPO compares the new and old policy on exactly
        the same stored action.
        """
        dist = self.distribution(obs)
        log_prob = self.joint_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(obs)

    def actor_parameters(self):
        yield from self.actor_mean.parameters()
        yield self.log_std

    def critic_parameters(self):
        yield from self.critic.parameters()

