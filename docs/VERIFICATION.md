# 本机验收记录

## JupyterLab 扩展验收（2026-11-15）

- 新增六本渐进式 Notebook，使用项目独立内核逐本从头运行。
- `python scripts/validate_notebooks.py`：六本全部通过，无 error 输出。
- 第 06 本完成 256 步短训练、CSV 读取、checkpoint 重载和两回合评估。
- Clip 正负优势图和短训练曲线已从保存输出提取并检查，标签完整、未裁切。
- 原有 7 项基础测试通过，`git diff --check` 通过。
- 保留已执行输出，便于在 GitHub 或 JupyterLab 中直接阅读。
- HTML 已导出；浏览器预览未能打开，因此未确认整页在浏览器中的显示。图形输出已单独查看。

重新执行：先按照 `notebooks/README.md` 安装并注册项目内核，再运行
`python scripts/validate_notebooks.py`。练习开关默认关闭，验收覆盖参考实现，
不表示尚未完成的学生 TODO 已通过。

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

