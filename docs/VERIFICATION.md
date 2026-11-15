# 本机验收记录

验收日期：2026-11-15（Asia/Shanghai）

## 环境

```text
Architecture: arm64 (Apple Silicon)
Python: 3.12.15
gymnasium: 1.4.0
mujoco: 4.0.1
numpy: 2.5.4
torch: 2.15.0
pytest: 9.2.0
```

这些是实际验收时解析到的版本，不是强制精确锁定。项目在 `pyproject.toml` 中使用兼容范围，避免把练习绑定到单个补丁版本。

## 测试结果

命令：

```bash
pytest
```

结果：

```text
7 passed in 15.70s
```

覆盖范围：

- `InvertedPendulum-v5` reset/step。
- 联合 Gaussian log probability 与张量形状。
- batch size 为 1 时 Actor/Critic 形状。
- `terminated` 不 bootstrap。
- `truncated` bootstrap 但阻断跨回合 GAE 递推。
- PPO 正负优势下的 clip 行为。
- Critic value loss 及梯度。

## 端到端短训练

命令：

```bash
python scripts/smoke_train.py
```

结果：

```text
step=    128 actor=-0.0022 critic=14.7217 episodes=14
step=    256 actor=-0.0015 critic=10.4264 episodes=27
smoke test passed: runs/smoke/final.pt
```

随后用 checkpoint 完成无渲染评估：

```text
mean_return=20.00 mean_length=21.0
```

这里只采样 256 步，结果用于验证完整执行链，不代表收敛性能。训练已成功经过：

```text
MuJoCo step -> rollout -> GAE/targets -> PPO mini-batch update
-> metrics.csv -> final.pt -> checkpoint reload -> evaluation
```

