"""Opt-in grader. Run: pytest -q checks/test_exercises.py"""

import torch
from torch.distributions import Normal

from ppo_pendulum.exercises.actor_critic import Critic, GaussianActor
from ppo_pendulum.exercises.gae import compute_gae
from ppo_pendulum.exercises.losses import ppo_clipped_actor_loss, value_loss


def test_stage_2_actor_and_critic() -> None:
    actor = GaussianActor(4, 2)
    dist = Normal(torch.zeros(3, 2), torch.ones(3, 2))
    actions = torch.zeros(3, 2)
    actual = actor.joint_log_prob(dist, actions)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, dist.log_prob(actions).sum(-1))
    assert Critic(4)(torch.zeros(1, 4)).shape == (1,)


def test_stage_3_gae_boundaries() -> None:
    advantages, targets = compute_gae(
        rewards=torch.tensor([1.0]),
        values=torch.tensor([0.25]),
        next_values=torch.tensor([4.0]),
        terminated=torch.tensor([False]),
        truncated=torch.tensor([True]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages, torch.tensor([4.35]))
    torch.testing.assert_close(targets, torch.tensor([4.6]))
    assert not advantages.requires_grad and not targets.requires_grad


def test_stage_4_losses() -> None:
    loss, ratio = ppo_clipped_actor_loss(
        torch.log(torch.tensor([1.5, 0.5])),
        torch.zeros(2),
        torch.tensor([1.0, -1.0]),
        0.2,
    )
    torch.testing.assert_close(ratio, torch.tensor([1.5, 0.5]))
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    predicted = torch.tensor([1.0], requires_grad=True)
    target = torch.tensor([3.0], requires_grad=True)
    critic_loss = value_loss(predicted, target)
    critic_loss.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([-2.0]))
    assert target.grad is None

