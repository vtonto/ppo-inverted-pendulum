# 手写 PPO：MuJoCo 倒立摆练习项目

这是一个面向控制工程师的渐进式练习项目。目标不是调用 Stable-Baselines3，而是亲手完成连续动作 PPO 的关键环节，并用 Gymnasium MuJoCo `InvertedPendulum-v5` 验证完整训练闭环。

项目提供两条可运行路径：

- `src/ppo_pendulum/exercises/`：按学习阶段拆分的完整实现，适合逐函数调试和修改。
- `src/ppo_pendulum/reference/`：结构更紧凑的对照实现，用于比较结果。

核心公式见 [docs/PPO_FORMULAS.md](docs/PPO_FORMULAS.md)，练习顺序见 [docs/EXERCISES.md](docs/EXERCISES.md)。

## 在 JupyterLab 中学习与调试

现在可以按顺序打开 [notebooks/](notebooks/README.md) 中的六本 Notebook。
Notebook 调用现有源码，适合打印张量、检查梯度、观察 GAE 边界和绘制 Clip/训练曲线。

激活项目虚拟环境后，从仓库根目录运行：

```bash
uv pip install -e ".[dev,notebook]"
python -m ipykernel install --sys-prefix --name ppo-pendulum --display-name "Python (PPO Pendulum)"
python -m jupyterlab
```

选择 **Python (PPO Pendulum)** 内核，从 `01_environment_basics.ipynb` 开始。
Notebook 直接执行 `exercises/` 中的完整实现，第 06 本可以跑完整训练、保存 checkpoint 并评估。
完整环境设置、源码自动重载、调试说明见 [Notebook 使用指南](notebooks/README.md)。

## 1. 环境与对象

`InvertedPendulum-v5` 的观测是 4 维向量，动作是 1 维连续力，动作范围为 `[-3, 3]`。每个健康时间步奖励为 1；杆角度或状态不健康会 `terminated=True`，到达时间上限会 `truncated=True`。

虽然这里的动作只有 1 维，代码始终保留 `[batch, action_dim]` 形式，因此可以自然迁移到多关节控制，并清楚看到联合概率密度中的 `.sum(-1)`。

官方说明：

- [Gymnasium InvertedPendulum-v5](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)
- [Gymnasium MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/)
- [PyTorch macOS installation](https://docs.pytorch.org/get-started/locally/)

## 2. Mac Apple Silicon 安装

推荐 Python 3.10–3.12。以下两种方式二选一。

### 使用 uv（推荐）

```bash
cd ppo-inverted-pendulum
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 使用标准 venv + pip

```bash
cd ppo-inverted-pendulum
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch 在这个小型 MLP 上使用 CPU 通常已经足够，也是本项目默认设置。若要尝试 Metal，可在训练命令中加 `--device mps`。

## 3. 三分钟验收

```bash
# 数学核心、张量形状与 MuJoCo 环境测试
pytest

# 256 步端到端短跑：只检查 rollout -> GAE -> update -> checkpoint
python scripts/smoke_train.py
```

短跑不会让策略收敛；它的验收标准只是：无异常退出，并生成：

```text
runs/smoke/config.json
runs/smoke/metrics.csv
runs/smoke/final.pt
```

## 4. 运行参考训练

运行按阶段拆分、便于调试的 exercise 完整实现：

```bash
ppo-train --implementation exercise --total-timesteps 20000
ppo-evaluate runs/exercise/final.pt --episodes 10
```

运行结构更紧凑的 reference 实现：

先做小规模观察：

```bash
ppo-train \
  --implementation reference \
  --total-timesteps 20000 \
  --rollout-steps 1024 \
  --minibatch-size 64 \
  --output-dir runs/first-run
```

再运行默认规模：

```bash
ppo-train --implementation reference --total-timesteps 200000
```

查看训练记录：

```bash
head runs/reference/metrics.csv
ppo-evaluate runs/reference/final.pt --episodes 10
```

需要打开 MuJoCo 窗口时：

```bash
ppo-evaluate runs/reference/final.pt --episodes 3 --render
```

倒立摆每回合默认最多 1000 步，所以评估时 `mean_length` 越接近 1000，说明策略越稳定。强化学习结果有随机性；不要把单个 seed 的一次训练当作严格结论。

## 5. 建议练习顺序

1. 环境交互：读懂 Gymnasium 的五元组返回值。
2. Gaussian Actor：实现联合 `log_prob` 和 Critic 输出形状。
3. GAE：分别处理 `terminated` 与 `truncated`。
4. PPO-Clip：实现 `ratio`、裁剪代理目标和 Critic loss。
5. Rollout：保存旧策略数据和 `V(s_{t+1})`。
6. Update：Mini-batch、多 Epoch、梯度裁剪与诊断指标。
7. 端到端接线：把练习实现接入 trainer，与参考版结果对比。

拆分函数可用独立检查器：

```bash
pytest -q checks/test_exercises.py
```

普通 `pytest` 验证参考实现与环境；上面的命令验证拆分后的 exercise 实现。

## 6. 工程目录

```text
ppo-inverted-pendulum/
├── README.md
├── pyproject.toml
├── docs/
│   ├── EXERCISES.md
│   └── PPO_FORMULAS.md
├── src/ppo_pendulum/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── exercises/
│   │   ├── actor_critic.py
│   │   ├── gae.py
│   │   ├── losses.py
│   │   ├── rollout.py
│   │   ├── update.py
│   │   └── trainer.py
│   └── reference/
│       ├── model.py
│       ├── algorithm.py
│       └── trainer.py
├── checks/test_exercises.py
├── notebooks/
│   ├── README.md
│   └── 01_...ipynb → 06_...ipynb
├── tests/
│   ├── test_reference_core.py
│   └── test_environment_smoke.py
└── scripts/smoke_train.py
```

## 7. 本实现刻意保持简单的地方

### Gaussian 动作与动作边界

Actor 使用未截断的因子化高斯分布。Rollout 保存原始采样 `raw_action` 及其 log probability，只把裁剪到 `[-3, 3]` 的动作送入环境。这样最容易学习联合 log probability，但严格说，裁剪后的环境动作分布不再是原高斯分布。

本环境动作范围较宽、初始标准差较小，这个近似通常足以教学。迁移到边界敏感任务时，应学习 `tanh` squashed Gaussian 及其 Jacobian 修正。

### Actor 与 Critic

两者使用分离网络和优化器，便于观察各自梯度。没有加入 value clipping、学习率退火、观测归一化、并行环境等增强项。

### MPS

训练默认 CPU。网络很小，环境仿真仍在 CPU，MPS 的数据搬运成本可能抵消收益。`--device mps` 是实验选项，不是必要条件。

## 8. 常见错误快速定位

- `log_prob` 形状还是 `[B, A]`：忘记沿动作维 `sum(-1)`。
- `ratio` 初始不接近 1：新旧策略没有评价同一批动作，或旧 log probability 被重算。
- Critic target 有梯度：GAE/target 没有 `detach()`。
- 时间截断的 target 偏低：把 `truncated` 错当成真正终止，丢失 bootstrap。
- GAE 跨回合传播：递推掩码没有同时在 `terminated` 和 `truncated` 处归零。
- batch size 为 1 时维度消失：用了不带维度参数的 `.squeeze()`，应使用 `.squeeze(-1)`。
- 先打乱再算 GAE：GAE 必须在时间有序的 rollout 上先计算，之后才能打乱做 mini-batch。

