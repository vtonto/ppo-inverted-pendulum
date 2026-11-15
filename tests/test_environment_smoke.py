import gymnasium as gym
import numpy as np


def test_inverted_pendulum_v5_reset_and_step() -> None:
    env = gym.make("InvertedPendulum-v5")
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (4,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()

