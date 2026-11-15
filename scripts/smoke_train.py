"""Short end-to-end run used for installation verification, not convergence."""

from pathlib import Path

from ppo_pendulum.config import PPOConfig
from ppo_pendulum.reference.trainer import train


if __name__ == "__main__":
    checkpoint = train(
        PPOConfig(
            total_timesteps=256,
            rollout_steps=128,
            update_epochs=2,
            minibatch_size=64,
            seed=7,
            output_dir=Path("runs/smoke"),
        )
    )
    print(f"smoke test passed: {checkpoint}")
