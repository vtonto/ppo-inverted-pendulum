# JupyterLab 学习入口

按 01–06 顺序打开 Notebook。每一本可在独立内核中从头运行，并直接调用 `src/ppo_pendulum/exercises/` 的完整实现。第 06 本会完整执行环境采样、GAE、PPO 更新、checkpoint 保存与评估。

## 安装与启动

在仓库根目录执行（已有 `.venv` 时跳过创建）：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev,notebook]"
python -m ipykernel install --sys-prefix --name ppo-pendulum --display-name "Python (PPO Pendulum)"
python -m jupyterlab
```

不使用 uv 时，先 `python3 -m venv .venv`，激活后用 `python -m pip install -e ".[dev,notebook]"`。

在浏览器左侧打开 `notebooks/01_environment_basics.ipynb`，右上角选择 **Python (PPO Pendulum)**。第一段输出的 Python 路径应该指向当前 `.venv`。Kernel 不同于浏览器：代码在所选 Kernel 的 Python 环境中执行，装错环境会导致模块找不到。

## 日常学习方式

1. 用 Shift+Enter 逐个执行 Cell，查看中间张量、图形和断言。
2. 双击左侧 `src/ppo_pendulum/exercises/*.py`，修改或设置断点并保存。
3. Notebook 已启用 `%autoreload 2`；修改源码后重新运行相关 Cell。修改类结构后，重新创建 Actor、Critic 或 Trainer 实例，必要时 Restart Kernel。
4. 运行每本中的检查 Cell；它们会验证当前内核使用的 exercise 实现。
5. 每完成一个阶段，使用 **Restart Kernel and Run All Cells**，确认没有依赖乱序执行产生的隐藏变量。

自动重载不会清空已有模型参数或优化器状态。不要在训练过程中修改网络结构后继续沿用旧优化器；重建实例再训练。

## 六本 Notebook

| 文件 | 主要观察对象 |
|---|---|
| 01_environment_basics | 观测、动作、reward、terminated/truncated |
| 02_gaussian_actor | `[B,A] -> [B]`、联合 log probability、梯度 |
| 03_gae_and_bootstrap | 两个 mask、逐步 delta、advantage 与 target |
| 04_ppo_clipped_loss | 正负 advantage、ratio、clip 曲线与 detach |
| 05_rollout_debugging | RolloutBatch、旧 log probability、更新前 ratio |
| 06_end_to_end_training | 256 步短训练、CSV 曲线、checkpoint 评估 |

第 06 本默认运行 256 步，只做管线检查，不代表训练收敛。训练结果写到仓库的 `runs/notebook/exercise/`，该目录已被 Git 忽略。重复执行会覆盖同名短训练结果；需要保留时修改 `output_dir`。

## 测试与调试

Notebook 内推荐用 `subprocess.run([sys.executable, '-m', 'pytest', ...])`，确保测试和当前 Kernel 使用同一个 Python。也可以在 JupyterLab 的 Terminal 中激活环境后运行 `pytest`。

可在一个短 Cell 中加入 `breakpoint()`，使用 IPython 的逐行调试；报错后运行 `%debug` 查看局部变量。JupyterLab 图形断点功能依赖 Kernel/调试器支持，普通 Cell 执行与 `%debug` 足以完成本练习。

默认不打开 MuJoCo 人机交互窗口，方便 Notebook 稳定执行。需要窗口时，可在终端使用现有 `ppo-evaluate ... --render` 命令。

公式详解见 [PPO_FORMULAS.md](../docs/PPO_FORMULAS.md)，逐阶段代码路线见 [EXERCISES.md](../docs/EXERCISES.md)。
