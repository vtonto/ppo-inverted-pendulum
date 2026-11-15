"""Stage 2: implement the factorized Gaussian actor.

Search for ``TODO(student)``. Compare with ``reference/model.py`` only after
your own implementation passes ``checks/test_exercises.py``.
"""

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
        # TODO(student): Normal.log_prob returns [batch, action_dim].
        # Sum only over the action dimension. Do not average across the batch.
        raise NotImplementedError("TODO: implement joint Gaussian log_prob")

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
        # TODO(student): remove only the final singleton dimension so that
        # output is [batch], including when batch_size == 1.
        raise NotImplementedError("TODO: implement Critic.forward")

