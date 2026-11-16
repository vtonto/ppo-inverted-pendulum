from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ppo_pendulum.reference.model import GaussianActorCritic


def evaluate(checkpoint_path: Path, episodes: int, render: bool) -> tuple[float, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    env = gym.make(
        config["env_id"],
        render_mode="human" if render else None,
        max_episode_steps=config["max_episode_steps"],
    )
    implementation = checkpoint.get("implementation", "reference")
    if implementation == "exercise":
        from ppo_pendulum.exercises.actor_critic import ExerciseActorCritic

        model = ExerciseActorCritic(
            checkpoint["obs_dim"], checkpoint["action_dim"], config["hidden_size"]
        )
    else:
        model = GaussianActorCritic(
            checkpoint["obs_dim"], checkpoint["action_dim"], config["hidden_size"]
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=config["seed"] + episode)
            episode_return = 0.0
            length = 0
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = model.deterministic_action(obs_t)[0]
                action = torch.clamp(
                    action,
                    torch.as_tensor(env.action_space.low),
                    torch.as_tensor(env.action_space.high),
                )
                obs, reward, terminated, truncated, _ = env.step(action.numpy())
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.mean(lengths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    mean_return, mean_length = evaluate(args.checkpoint, args.episodes, args.render)
    print(f"mean_return={mean_return:.2f} mean_length={mean_length:.1f}")


if __name__ == "__main__":
    main()
