# PPO 原理与实现：关键公式、推导与代码参考

> 面向连续动作控制的独立复习文档。符号尽量沿用赵世钰《Mathematical Foundations of Reinforcement Learning》中的 (v_\pi(s))、(q_\pi(s,a)) 与初始状态价值指标，再补充 GAE 和 PPO 的工程记号。

## 0. 阅读地图

完整数据流是：

```text
Actor 根据 s_t 产生分布
        ↓ 采样 a_t，同时保存 old_log_prob_t
环境执行动作，返回 r_t, s_{t+1}, terminated, truncated
        ↓
Critic 给出 V(s_t), V(s_{t+1})
        ↓ 时间有序地倒推
TD error -> GAE advantage -> Critic target
        ↓ 打乱成 mini-batch，多 epoch
PPO clipped Actor update + value regression Critic update
        ↓
丢弃旧 rollout，用新策略重新采样
```

Actor 回答“在状态 (s) 下采用什么动作”；Critic 估计当前策略的状态价值。PPO 是 on-policy 风格算法：旧 rollout 可以重复若干 epoch，但不能无限复用。

---

## 1. 优化对象、价值函数与策略梯度

### 1.1 价值函数

对策略 (pi)：

\[
v_\pi(s)
=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\mid S_t=s\right]
\]

\[
q_\pi(s,a)
=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\mid S_t=s,A_t=a\right]
\]

优势函数表示动作相对状态平均水平的增益：

\[
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)
\]

### 1.2 初始状态价值指标

若初始状态按 (d_0(s)) 分布，常见优化目标可写为：

\[
\bar v_\pi^0
=\mathbb E_{S_0\sim d_0}[v_\pi(S_0)]
=\sum_s d_0(s)v_\pi(s)
\]

(d_0) 是“回合从哪里开始”的分布；策略梯度定理中的折扣状态访问权重 (ho_\pi(s)) 描述策略运行后各状态在目标中的权重。两者不是同一个量。

工程 PPO 通常用有限 rollout 的样本均值近似这些期望，并不会显式计算完整的 (ho_\pi(s))。这是理论目标到采样算法的近似桥梁。

### 1.3 对数导数技巧

利用：

\[
\nabla_\theta \pi_\theta(a\mid s)
=\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)
\]

策略梯度可写成统一形式：

\[
\nabla_\theta J(\theta)
\propto
\mathbb E_{s,a\sim\pi_\theta}
\left[
Q(s,a)\nabla_\theta\log\pi_\theta(a\mid s)
\right]
\]

加入只依赖状态的 baseline 不改变期望梯度，于是可用优势：

\[
\nabla_\theta J(\theta)
\propto
\mathbb E
\left[
A_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)
\right]
\]

直觉：(hat A_t>0) 时提高已采样动作的密度，(hat A_t<0) 时降低它的密度。

---

## 2. 连续动作 Gaussian Actor

### 2.1 Actor 输出分布，不直接输出唯一动作

对 (m) 维动作，因子化高斯策略输出：

\[
\boldsymbol\mu_\theta(s)\in\mathbb R^m,
\qquad
\boldsymbol\sigma_\theta(s)>0
\]

并采样：

\[
a_i\sim\mathcal N(\mu_i(s),\sigma_i^2(s))
\]

本项目的均值由神经网络给出，`log_std` 是可学习参数：

```python
mean = actor_mean(obs)                  # [B, A]
std = log_std.exp().expand_as(mean)     # [B, A]
dist = torch.distributions.Normal(mean, std)
raw_action = dist.sample()              # [B, A]
```

### 2.2 为什么是联合概率密度

机器人在一个时刻执行的是整个动作向量：

\[
\boldsymbol a=(a_1,\ldots,a_m)
\]

所以策略梯度中的 (pi_\theta(a\mid s)) 是完整动作向量的联合密度。常见对角高斯假设给定状态后各维采样噪声条件独立：

\[
\pi_\theta(\boldsymbol a\mid s)
=\prod_{i=1}^{m}\pi_{\theta,i}(a_i\mid s)
\]

这不表示关节动力学独立。共享网络仍可根据完整状态协调各维均值；独立的是给定状态后的采样噪声。

### 2.3 `.sum(-1)` 的数学含义

乘积取对数变成求和：

\[
\log\pi_\theta(\boldsymbol a\mid s)
=\sum_{i=1}^{m}\log\pi_{\theta,i}(a_i\mid s)
\]

代码：

```python
per_dim = dist.log_prob(actions)  # [B, A]
joint = per_dim.sum(dim=-1)       # [B]
```

只沿最后的动作维求和，绝不能沿 batch 维求和。每条样本最终有一个联合 log probability，对应一个优势 (hat A_t)。

即使 InvertedPendulum 的 (A=1)，也保留 `.sum(-1)`，使语义明确并能直接迁移到多关节任务。

### 2.4 不需要真的乘出很小的密度

PPO 用 log probability 的差：

\[
r_t(\theta)
=\frac{\pi_\theta(\boldsymbol a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(\boldsymbol a_t\mid s_t)}
=\exp\!\left(
\log\pi_\theta(\boldsymbol a_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(\boldsymbol a_t\mid s_t)
\right)
\]

```python
new_log_prob = new_dist.log_prob(stored_actions).sum(-1)
ratio = torch.exp(new_log_prob - old_log_prob)
```

这样比先乘出联合密度再相除更稳定。

---

## 3. Critic、TD error 与 GAE

### 3.1 一步 TD error

Critic 用 (V_\phi(s)) 近似 (v_\pi(s))。一步 TD error：

\[
\delta_t
=r_t+\gamma b_t V_\phi(s_{t+1})-V_\phi(s_t)
\]

其中 (b_t) 是 bootstrap mask。正常 transition 和时间截断允许 bootstrap，真正终止不允许：

\[
b_t=1-\mathbb 1[\text{terminated}_t]
\]

在价值估计足够准确时，TD error 可作为优势的有噪声估计。

### 3.2 GAE

GAE 把多步 TD error 指数加权：

\[
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\delta_t+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots
\]

递推形式：

\[
\hat A_t
=\delta_t+\gamma\lambda c_t\hat A_{t+1}
\]

这里递推 mask 必须在任何 episode 边界停止：

\[
c_t=1-\mathbb 1[\text{terminated}_t\lor\text{truncated}_t]
\]

这两个 mask 的区别非常重要：

| 情况 | (delta_t) 中使用 (V(s_{t+1})) | GAE 递推到下一条数组数据 |
|---|---:|---:|
| 普通 transition | 是 | 是 |
| `terminated=True` | 否 | 否 |
| `truncated=True` | 是 | 否 |

时间截断并不是 MDP 的吸收终态，因此 target 仍应利用截断前 `next_obs` 的价值；但 reset 后数组中的下一条数据属于新回合，优势不能跨过去。

### 3.3 (gamma) 与 (lambda)

- (gamma)：任务对未来奖励的折扣。
- (lambda=0)：接近一步 TD，方差小、偏差较大。
- (lambda\to1)：包含更长未来信息，偏差小、方差较大。
- 常用起点：(gamma=0.99,lambda=0.95)。

### 3.4 Critic target

本项目采用：

\[
y_t=V_{\mathrm{old}}(s_t)+\hat A_t
\]

Critic 损失：

\[
L_V(\phi)
=\frac12\mathbb E_t[(V_\phi(s_t)-y_t)^2]
\]

代码对应：

```python
returns = (advantages + old_values).detach()  # [T]
critic_loss = 0.5 * ((new_values - returns) ** 2).mean()
```

(hat A_t) 同时有两个用途：作为 Actor 更新的权重；与旧 value 相加构造 Critic target。用途不同，但都来自同一 rollout 的回报信息。

---

## 4. PPO 的 ratio 与 Clip

### 4.1 为什么需要新旧策略比

rollout 是由旧参数 (	heta_{\mathrm{old}}) 采集的。PPO 在同一批数据上更新多次，用重要性比率近似校正策略变化：

\[
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\]

第一次 update 前，新旧参数相同，所以同一动作的 ratio 应接近 1。若不是，通常是动作、log probability 或旧参数处理错了。

### 4.2 Clipped surrogate objective

\[
L^{\mathrm{CLIP}}(\theta)
=\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
\]

优化器做最小化，所以代码返回负号：

```python
ratio = torch.exp(new_log_prob - old_log_prob)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
actor_loss = -torch.minimum(surr1, surr2).mean()
```

### 4.3 正负优势的非对称直觉

当 (hat A_t>0)：该动作比平均好，想提高其密度；ratio 超过 (1+\epsilon) 后收益被截平。

当 (hat A_t<0)：该动作比平均差，想降低其密度；ratio 低于 (1-\epsilon) 后收益被截平。

`min` 与优势符号共同产生这个方向。不要把它改成对两个 surrogate 的平均。

### 4.4 Clip 的边界

Clip 约束的是样本上的 probability ratio 对目标函数的影响，不是对所有状态的参数距离进行严格限制。它也不能保证每一步 KL 都小，因此工程实现常同时监控 approximate KL 与 clip fraction。

---

## 5. 完整损失与梯度流

### 5.1 Actor

若加入 entropy bonus：

\[
L_{\mathrm{actor}}
=-L^{\mathrm{CLIP}}(\theta)
-c_H\,\mathbb E_t[\mathcal H(\pi_\theta(\cdot\mid s_t))]
\]

本项目默认 (c_H=0)，但仍记录 entropy。

### 5.2 Critic

\[
L_{\mathrm{critic}}
=c_V\frac12\mathbb E_t[(V_\phi(s_t)-y_t)^2]
\]

### 5.3 哪些量需要梯度

| 量 | 来自哪里 | update 时需要梯度？ |
|---|---|---:|
| `new_log_prob` | 当前 Actor 重算 | 是 |
| `predicted_value` | 当前 Critic 重算 | 是 |
| `old_log_prob` | rollout 保存 | 否 |
| `old_value` | rollout 保存 | 否 |
| `advantage` | GAE | 否 |
| `critic_target` | GAE + old value | 否 |
| `stored_action` | rollout 采样 | 否 |

`detach()` 的目的不是“提高稳定性”这样笼统，而是明确训练目标：更新 Actor 时不反向穿过采样时的旧策略；更新 Critic 时 target 是监督标签，不能随当前预测一起移动。

```python
advantages = advantages.detach()
returns = returns.detach()
old_log_probs = old_log_probs.detach()
```

---

## 6. Rollout 数据结构与形状

令 rollout 长度为 (T)，mini-batch 大小为 (B)，观测维度 (S)，动作维度 (A)：

| 名称 | Rollout 形状 | Mini-batch 形状 | 含义 |
|---|---:|---:|---|
| observations | `[T, S]` | `[B, S]` | (s_t) |
| raw_actions | `[T, A]` | `[B, A]` | 高斯采样动作 |
| old_log_probs | `[T]` | `[B]` | 完整动作联合 log density |
| rewards | `[T]` | 通常不直接更新 | (r_t) |
| old_values | `[T]` | `[B]` | rollout 时 (V(s_t)) |
| next_values | `[T]` | GAE 前使用 | rollout 时 (V(s_{t+1})) |
| terminated | `[T]` | GAE 前使用 | 真正 MDP 终止 |
| truncated | `[T]` | GAE 前使用 | 外部时间限制 |
| advantages | `[T]` | `[B]` | (hat A_t) |
| returns/targets | `[T]` | `[B]` | (y_t) |

GAE 必须先在时间有序的 `[T]` 数据上计算。完成 GAE 后，Actor/Critic 的监督样本才可以打乱成 mini-batch。

---

## 7. 完整训练伪代码

```python
initialize actor theta, critic phi
obs = env.reset()

while global_step < total_timesteps:
    rollout = []

    for t in range(T):
        dist = actor(obs)
        raw_action = dist.sample()
        old_log_prob = dist.log_prob(raw_action).sum(-1)
        old_value = critic(obs)

        env_action = clip_to_action_space(raw_action)
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        next_value = critic(next_obs)  # reset 前计算

        rollout.append(
            obs, raw_action, old_log_prob, reward,
            old_value, next_value, terminated, truncated
        )

        if terminated or truncated:
            obs = env.reset()
        else:
            obs = next_obs

    advantages, targets = compute_gae(rollout)
    advantages = normalize(advantages)

    for epoch in range(K):
        for minibatch in shuffled_minibatches(rollout):
            new_dist = actor(minibatch.obs)
            new_log_prob = new_dist.log_prob(minibatch.raw_action).sum(-1)
            ratio = exp(new_log_prob - minibatch.old_log_prob)

            actor_loss = negative_clipped_surrogate(ratio, minibatch.advantage)
            critic_loss = half_mse(critic(minibatch.obs), minibatch.target)

            update_actor(actor_loss)
            update_critic(critic_loss)
```

关键点：更新时评价的是 rollout 中同一组旧动作，不是重新采样动作；否则 ratio 不再对应同一个事件。

---

## 8. `terminated`、`truncated` 与 rollout 截断

需要区分三种边界：

1. **环境真终止**：例如杆倒下，未来任务回报结束，不 bootstrap。
2. **TimeLimit 截断**：任务本身未终止，只是外部限制，使用截断前 next state bootstrap。
3. **rollout 恰好收集到 T 步**：环境还在继续，下次 rollout 接着同一 episode；正常 bootstrap，GAE 在当前数组末尾自然停止。

本项目为每个 transition 直接保存 `next_value`，因此不会依赖“数组下一行恰好是哪个 episode”。这比仅保存一个 rollout 末尾 value 更啰嗦，但对学习边界语义更透明。

---

## 9. 实现检查清单

### Actor

- [ ] `Normal.log_prob(actions)` 是 `[B, A]`。
- [ ] `.sum(-1)` 后是 `[B]`。
- [ ] 新旧策略评价同一 `stored_actions`。
- [ ] `old_log_prob` 在 rollout 时保存，update 时固定。
- [ ] batch size 为 1 时没有误用无参数 `.squeeze()`。

### GAE 与 Critic

- [ ] 倒序递推。
- [ ] `terminated` 关闭 bootstrap。
- [ ] `terminated or truncated` 关闭跨 episode 递推。
- [ ] `next_value` 在 reset 前计算。
- [ ] target 为 `old_value + advantage`。
- [ ] advantage 与 target 已 detach。

### PPO update

- [ ] `ratio = exp(new_log_prob - old_log_prob)`。
- [ ] Actor 最大化目标在代码中转成负 loss 最小化。
- [ ] 每个 epoch 重新打乱索引。
- [ ] GAE 之前不打乱时间顺序。
- [ ] 使用梯度裁剪。
- [ ] 监控 approximate KL、clip fraction、entropy。

### 调试信号

- [ ] 首次 update 前 ratio 接近 1。
- [ ] 所有 loss 有限，无 NaN/Inf。
- [ ] Critic target 与 value 量级合理。
- [ ] episode length/return 随训练总体改善。
- [ ] 至少用多个 seed 复核，不只挑最好的一次。

---

## 10. 公式速查

\[
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)
\]

\[
\delta_t=r_t+\gamma(1-\text{terminated}_t)V(s_{t+1})-V(s_t)
\]

\[
\hat A_t=\delta_t+\gamma\lambda
(1-(\text{terminated}_t\lor\text{truncated}_t))\hat A_{t+1}
\]

\[
y_t=V_{\mathrm{old}}(s_t)+\hat A_t
\]

\[
\log\pi_\theta(\boldsymbol a\mid s)
=\sum_i\log\pi_{\theta,i}(a_i\mid s)
\]

\[
r_t(\theta)=\exp(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t))
\]

\[
L^{\mathrm{CLIP}}
=\mathbb E_t[
\min(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]
\]

\[
L_V=\frac12\mathbb E_t[(V_\phi(s_t)-y_t)^2]
\]

最后应记住的主线：**旧策略采样并固定训练数据；GAE 将奖励和 Critic 估计变成 advantage/target；当前策略用同一动作重算联合 log probability；ratio 与 clip 限制单批数据上的更新幅度；Actor 与 Critic 各自只沿正确的梯度路径更新。**

