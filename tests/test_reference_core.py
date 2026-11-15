import torch
from torch.distributions import Normal

from ppo_pendulum.reference.algorithm import (
    compute_gae,
    ppo_clipped_actor_loss,
    value_loss,
)
from ppo_pendulum.reference.model import GaussianActorCritic


def test_joint_log_prob_sums_action_dimension_only() -> None:
    dist = Normal(torch.zeros(3, 2), torch.ones(3, 2))
    actions = torch.tensor([[0.0, 0.0], [1.0, -1.0], [0.5, 0.25]])
    actual = GaussianActorCritic.joint_log_prob(dist, actions)
    expected = dist.log_prob(actions).sum(dim=-1)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected)


def test_model_shapes_for_batch_of_one() -> None:
    model = GaussianActorCritic(obs_dim=4, action_dim=1)
    obs = torch.zeros(1, 4)
    action, log_prob, value = model.act(obs)
    assert action.shape == (1, 1)
    assert log_prob.shape == (1,)
    assert value.shape == (1,)


def test_gae_terminated_removes_bootstrap() -> None:
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.25])
    next_values = torch.tensor([10.0])
    advantages, targets = compute_gae(
        rewards,
        values,
        next_values,
        terminated=torch.tensor([True]),
        truncated=torch.tensor([False]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages, torch.tensor([0.75]))
    torch.testing.assert_close(targets, torch.tensor([1.0]))


def test_gae_truncation_bootstraps_but_stops_recursion() -> None:
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 0.25])
    next_values = torch.tensor([4.0, 7.0])
    advantages, targets = compute_gae(
        rewards,
        values,
        next_values,
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([True, False]),
        gamma=0.9,
        gae_lambda=1.0,
    )
    # t=0 is truncated: delta includes 0.9*4, but recursion cannot include t=1.
    torch.testing.assert_close(advantages[0], torch.tensor(4.1))
    torch.testing.assert_close(targets[0], torch.tensor(4.6))


def test_ppo_clip_for_positive_and_negative_advantages() -> None:
    old = torch.zeros(2)
    new = torch.log(torch.tensor([1.5, 1.5]))
    advantages = torch.tensor([1.0, -1.0])
    loss, ratio, clip_fraction = ppo_clipped_actor_loss(new, old, advantages, 0.2)
    torch.testing.assert_close(ratio, torch.tensor([1.5, 1.5]))
    # min(1.5, 1.2)=1.2 for A>0; min(-1.5, -1.2)=-1.5 for A<0.
    torch.testing.assert_close(loss, torch.tensor(0.15))
    torch.testing.assert_close(clip_fraction, torch.tensor(1.0))


def test_value_loss_and_gradient() -> None:
    predicted = torch.tensor([1.0, 3.0], requires_grad=True)
    targets = torch.tensor([2.0, 1.0])
    loss = value_loss(predicted, targets)
    torch.testing.assert_close(loss, torch.tensor(1.25))
    loss.backward()
    torch.testing.assert_close(predicted.grad, torch.tensor([-0.5, 1.0]))

