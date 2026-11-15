from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ppo_pendulum.config import PPOConfig
from ppo_pendulum.reference.trainer import train


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train from-scratch PPO on InvertedPendulum-v5")
    parser.add_argument("--implementation", choices=["reference", "exercise"], default="reference")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2_048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reference"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.implementation == "exercise":
        raise SystemExit(
            "练习实现包含有意保留的 TODO。请先按 docs/EXERCISES.md 完成各阶段，"
            "再将其中函数接入 trainer；端到端验收请使用 --implementation reference。"
        )
    config = PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        device=_resolve_device(args.device),
        output_dir=args.output_dir,
    )
    checkpoint = train(config)
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

