# 分阶段练习与验收

这份路线的原则是：每次只手写一个数学环节，先用小张量验收，再接入环境。不要一开始就盯着训练曲线调参；数学核心没通过单元测试时，长训练只会放大定位难度。

## 阶段 0：建立基线

目标：确认 Python、PyTorch、Gymnasium 和 MuJoCo 可用。

```bash
pytest
python scripts/smoke_train.py
```

验收：普通测试全通过，短训练生成 `runs/smoke/final.pt`。这里跑的是参考版，不要求学习收敛。

## 阶段 1：只做环境交互

新建一个临时脚本或在 Python REPL 中完成：

```python
import gymnasium as gym

env = gym.make("InvertedPendulum-v5")
obs, info = env.reset(seed=0)
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

需要回答：

1. `obs`、`action` 的形状是什么？
2. `terminated` 与 `truncated` 各代表什么？
3. 哪个状态用于计算 `V(s_{t+1})`，为什么要在 reset 之前计算？

验收：能打印每个值的类型和形状，能在 `terminated or truncated` 时正确 reset。

## 阶段 2：Gaussian Actor 与 Critic

编辑：

- `src/ppo_pendulum/exercises/actor_critic.py`

完成：

- `GaussianActor.joint_log_prob`
- `Critic.forward`

关键形状：

```text
obs                    [B, obs_dim]
mean, std              [B, action_dim]
dist.log_prob(actions) [B, action_dim]
joint_log_prob         [B]
critic value           [B]
```

运行：

```bash
pytest -q checks/test_exercises.py -k stage_2
```

思考：InvertedPendulum 的 `action_dim=1` 时，`.sum(-1)` 看起来没有改变数值。为什么仍要保留它？把动作维临时改成 2，再观察张量形状。

## 阶段 3：GAE 与 Critic targets

编辑：

- `src/ppo_pendulum/exercises/gae.py`

完成倒序递推：

```text
delta_t -> A_hat_t -> y_t
```

必须分别构造两个 mask：

- `bootstrap_mask = 1 - terminated`
- `recursion_mask = 1 - (terminated or truncated)`

运行：

```bash
pytest -q checks/test_exercises.py -k stage_3
```

手算验收：单步 transition 中，`r=1, V(s)=0.25, V(s')=4, gamma=0.9`：

- 真终止：`delta = 1 - 0.25 = 0.75`，target 为 `1`。
- 时间截断：`delta = 1 + 0.9*4 - 0.25 = 4.35`，target 为 `4.6`。

## 阶段 4：PPO-Clip 与 Critic loss

编辑：

- `src/ppo_pendulum/exercises/losses.py`

完成：

- `ratio = exp(new_log_prob - old_log_prob)`
- unclipped 与 clipped surrogate
- 最小化形式的 Actor loss
- detached target 的 Critic loss

运行：

```bash
pytest -q checks/test_exercises.py -k stage_4
```

建议额外打印 `ratio`。在首次 update、参数还未更新时，同一批动作的新旧 log probability 应相等，因此 ratio 应接近 1。

## 阶段 5：Rollout

编辑：

- `src/ppo_pendulum/exercises/rollout.py`

推荐先写出一个字典列表，再优化成预分配张量。每条 transition 至少保存：

```text
obs_t, raw_action_t, old_log_prob_t, reward_t,
value_t, next_value_t, terminated_t, truncated_t
```

验收不看学习结果，先断言：

```python
assert observations.shape == (T, obs_dim)
assert raw_actions.shape == (T, action_dim)
assert old_log_probs.shape == (T,)
assert rewards.shape == (T,)
assert values.shape == (T,)
```

再人工制造一次短时间限制，确认：

- 截断 transition 的 `next_value` 在 reset 前得到；
- GAE 不跨 reset 后的新 episode 传播。

## 阶段 6：Update

编辑：

- `src/ppo_pendulum/exercises/update.py`

推荐按以下顺序写：

1. 整个 rollout 上归一化 advantage。
2. 每个 epoch 生成一组随机索引。
3. 用当前 Actor 对存下来的动作重新求 `new_log_prob`。
4. 用固定的 `old_log_prob` 与 advantage 计算 Actor loss。
5. 用固定 target 回归 Critic。
6. 分别 `zero_grad -> backward -> clip_grad_norm -> step`。
7. 记录 KL 近似、clip fraction、entropy。

验收：

- 第一个 mini-batch 更新前 ratio 均值接近 1。
- `old_log_probs.requires_grad == False`。
- `advantages.requires_grad == False`。
- `returns.requires_grad == False`。
- Actor loss 只改变 Actor 参数；Critic loss 只改变 Critic 参数。

## 阶段 7：端到端接线

最简单的方法是复制 `reference/trainer.py` 为自己的 `student_trainer.py`，然后逐个将 import 替换为练习实现。这样可以保留参考版作为随时可运行的基线。

先用短配置：

```text
total_timesteps = 256
rollout_steps = 128
update_epochs = 2
minibatch_size = 64
```

通过后再改为 20,000 和 200,000 步。每次只改一个因素，并保留 `metrics.csv`。

## 阶段 8：有余力再做的扩展

按建议顺序：

1. 观测归一化。
2. 学习率线性退火。
3. value clipping。
4. KL 阈值提前停止 update epoch。
5. 并行环境。
6. tanh-squashed Gaussian 及 Jacobian 修正。
7. 多随机种子统计，而不是只看一条训练曲线。

