from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class PPOConfig:
    """All hyperparameters used by the reference trainer."""

    env_id: str = "InvertedPendulum-v5"
    total_timesteps: int = 200_000
    rollout_steps: int = 2_048
    update_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    seed: int = 1
    device: str = "cpu"
    max_episode_steps: int = 1_000
    output_dir: Path = Path("runs/reference")

    def validate(self) -> None:
        if self.total_timesteps <= 0 or self.rollout_steps <= 0:
            raise ValueError("total_timesteps and rollout_steps must be positive")
        if self.minibatch_size <= 0 or self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size must be in [1, rollout_steps]")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.clip_coef <= 0.0:
            raise ValueError("clip_coef must be positive")

    def as_serializable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        return result

