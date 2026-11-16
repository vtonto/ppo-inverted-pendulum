"""Checks for the readable exercise implementation.

Run: pytest -q checks/test_exercises.py
"""

import numpy as np
import torch
from torch.distributions import Normal

from ppo_pendulum.exercises.actor_critic import Critic, GaussianActor
from ppo_pendulum.exercises.gae import compute_gae
from ppo_pendulum.exercises.losses import ppo_clipped_actor_loss, value_loss
from ppo_pendulum.config import PPOConfig
from ppo_pendulum.exercises.trainer import ExercisePPOTrainer


def test_stage_2_actor_and_critic() -> None:
    actor = GaussianActor(4, 2)
    dist = Normal(torch.zeros(3, 2), torch.ones(3, 2))
    actions = torch.zeros(3, 2)
    actual = actor.joint_log_prob(dist, actions)
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, dist.log_prob(actions).sum(-1))
    assert Critic(4)(torch.zeros(1, 4)).shape == (1,)


def test_stage_3_gae_boundaries() -> None:
    values = torch.tensor([0.25], requires_grad=True)
    next_values = torch.tensor([4.0], requires_grad=True)
    advantages, targets = compute_gae(
        rewards=torch.tensor([1.0]),
        values=values,
        next_values=next_values,
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


def test_stage_5_rollout_is_detached_and_ratio_starts_at_one() -> None:
    config = PPOConfig(
        total_timesteps=32,
        rollout_steps=32,
        minibatch_size=16,
        update_epochs=1,
        max_episode_steps=4,
        seed=7,
    )
    trainer = ExercisePPOTrainer(config)
    try:
        batch = trainer.collect_rollout(32)
        assert batch.observations.shape == (32, 4)
        assert batch.raw_actions.shape == (32, 1)
        assert not any(
            tensor.requires_grad
            for tensor in (
                batch.observations,
                batch.raw_actions,
                batch.old_log_probs,
                batch.advantages,
                batch.returns,
                batch.values,
            )
        )
        with torch.no_grad():
            new_log_prob, _, _ = trainer.model.evaluate_actions(
                batch.observations, batch.raw_actions
            )
        torch.testing.assert_close(
            (new_log_prob - batch.old_log_probs).exp(), torch.ones(32)
        )
        assert trainer.completed_lengths
        assert max(trainer.completed_lengths) <= 4
    finally:
        trainer.close()


def test_stage_6_update_changes_actor_and_preserves_rollout() -> None:
    config = PPOConfig(
        total_timesteps=64,
        rollout_steps=64,
        minibatch_size=32,
        update_epochs=2,
        seed=7,
    )
    trainer = ExercisePPOTrainer(config)
    try:
        batch = trainer.collect_rollout(64)
        old_log_probs = batch.old_log_probs.clone()
        actor_before = [parameter.detach().clone() for parameter in trainer.model.actor_parameters()]
        stats = trainer.update(batch)
        torch.testing.assert_close(batch.old_log_probs, old_log_probs)
        assert all(np.isfinite(value) for value in stats.values())
        actor_after = list(trainer.model.actor_parameters())
        assert any(
            not torch.equal(before, after.detach())
            for before, after in zip(actor_before, actor_after)
        )
    finally:
        trainer.close()

