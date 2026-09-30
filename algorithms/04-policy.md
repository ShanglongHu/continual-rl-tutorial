[学习首页](../README.md) · [算法与代码目录](README.md)

# 04 直接改进策略：REINFORCE、actor–critic、GAE 与 PPO

先修：TD；概率、对数与梯度

问题：不先对每个动作求最优 Q，能不能直接增加好动作出现的概率？

REINFORCE → baseline / actor–critic → GAE → TRPO → PPO

参数化策略 πθ 直接给出动作分布。核心问题变为：哪次行动让回报高于预期，就适当提高它在相似状态下的概率。先理解这个信用分配，再看 PPO 的裁剪；从复杂 loss 倒着背，往往会漏掉每个量从哪里来。

符号：τ：轨迹；J：期望回报；θ：策略参数；Vφ：critic；Â：估计优势；ρ：新旧动作概率比。这里先用有限 episode、γ=1 推导简单策略梯度。

## 为什么出现 log π

R(τ) 是整条轨迹回报，Gₜ 只保留从 t 起的奖励。轨迹概率由初始状态概率、策略概率与环境转移概率相乘。假设环境本身不依赖 θ，用 ∇p=p∇log p，将期望回报的梯度写成能用轨迹采样估计的形式。过去已发生的奖励不应给未来动作分信用，因此可以用 reward-to-go。

$$
\begin{aligned}\nabla_\theta J&=\sum_\tau p_\theta(\tau)R(\tau)\nabla_\theta\log p_\theta(\tau)\\\nabla_\theta\log p_\theta(\tau)&=\sum_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\\\nabla_\theta J&=\mathbb E\!\left[\sum_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)G_t\right]\end{aligned}
$$

## 减去 baseline，再用 critic 近似

仅由状态决定、且在 actor 更新时停止梯度的 baseline，不改变期望策略梯度；它可减少方差。Actor–critic 再用价值函数自举，换来更及时的更新，也可能引入估计偏差。

$$
\mathbb E_{A\sim\pi}[\nabla_\theta\log\pi_\theta(A\mid s)b(s)]=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0,\qquad \widehat A_t\approx\delta_t
$$

## GAE：把不同距离的 TD 误差组合起来

GAE 用衰减系数组织多个 TD residual。λ 控制偏差与方差的折中，不能凭“越长越好”选择。有限 rollout 末端还要处理 bootstrap 与终止。

$$
\delta_t=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t),\qquad \widehat A_t^{\mathrm{GAE}}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}
$$

## PPO：限制同一批数据上继续放大更新的激励

同一批 rollout 被多次优化后，策略已偏离采样策略。PPO-Clip 用概率比和裁剪后的保守目标控制更新；它不是严格的 KL 约束，也不保证每个动作概率都落在裁剪区间。

$$
\rho_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\mathrm{old}}(A_t\mid S_t)},\quad L^{\mathrm{clip}}=\mathbb E\left[\min\big(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\big)\right]
$$

## 手算例子

两个动作概率均为 0.5，选中第二个且优势为 +1。softmax logits 的梯度是 (−0.5,+0.5)；步长 0.1 后 logits 从 (0,0) 变成 (−0.05,+0.05)，第二动作概率约 0.525。PPO 中若优势 +2、ρ=1.4、ε=0.2，该样本裁剪目标取 min(2.8,2.4)=2.4。

## 对照代码

```python
# 表格 softmax 的 log-prob 梯度；policy_step() 的实际更新
for j in range(len(logits)):
    logits[j] += alpha * advantage * ((j == action) - probabilities[j])
# REINFORCE: advantage = reward-to-go（此处不加 baseline）
# 一步 actor–critic: advantage = reward + V(next_state) - V(state)
```

标准库版 policy() 比较 REINFORCE 与一步 actor–critic，不实现 PPO。先定位采样时缓存的概率与更新时的梯度，再进入 CleanRL ppo.py，按 rollout → advantages → minibatches → losses → optimizer 的顺序读。

```sh
python3 examples/rl_foundations.py policy --seeds 5 --episodes 800 --alpha 0.05 --switch 400 --out policy-run
```

看回报与起点向右的概率，而不只看 actor loss。环境变了以后，过于确定的策略可能很难再采到另一侧；这个现象不等同于深度网络可塑性损失。

## 怎样接到 CRL

PPO 是常用的 CRL 实验底座，但其批量 rollout 与多轮更新不自动满足严格 streaming。Critic、策略、归一化和优化器的长期状态，都可能影响后续适应。

适用条件：本章基础梯度推导使用 γ=1；折扣任务需统一状态访问权重与目标定义。用近似 critic、有限 GAE 或 PPO 裁剪后的优化，不再是同一个无偏 MC 梯度估计器。

## 思考题

优势为 −2、ρ=0.6、ε=0.2，PPO-Clip 为什么取 −1.6 而不是 −1.2？

提示：先计算两项，再取 min；不要把负优势按正优势理解。

参考答案：ρÂ=−1.2，clip(ρ)Â=0.8×(−2)=−1.6。目标不再鼓励继续把这个负优势动作的概率过度降低。

## 原始材料与代码

- [Spinning Up · 策略梯度推导与 PyTorch 小实现](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：从轨迹概率开始，再对照代码。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：读多步 TD residual 的组合。
- [PPO 讲解](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：比较正、负优势的裁剪。
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)：追完整 rollout 与优化循环。


---

[← 上一章](03-deep-value.md) · [下一章 →](05-soft-control.md)
