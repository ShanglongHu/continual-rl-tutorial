[学习首页](../README.md) · [算法与代码目录](README.md)

# 05 连续动作与数据复用：DDPG、TD3、SAC

先修：Q-learning、actor–critic、函数梯度

问题：动作是连续扭矩，无法枚举所有 a 取 max，怎么办？

连续动作的 max 难题 → DDPG → TD3 → SAC

可以学一个 actor 来近似挑出高 Q 的连续动作，同时用 Bellman target 训练 critic。这样把价值学习与策略优化接在一起。TD3 和 SAC 并非把 PPO 简单换一个损失，而是具有不同数据与更新协议的 off-policy actor–critic。

符号：μθ：确定性 actor；πθ：随机 actor；Qφ₁,Qφ₂：两个 critic；η：熵温度（避免与学习步长 α 混用）；D：回放数据。

## DDPG：让 actor 沿 critic 指出的方向走

固定 critic 时，actor 希望选出的动作有更高 Q。链式法则把 Q 对动作的梯度传回 actor。Critic 错误也可能被 actor 利用，因此 Q 的拟合误差不是单纯的预测误差。

$$
\nabla_\theta J\approx\mathbb E_{s\sim D}\!\left[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\;\nabla_\theta\mu_\theta(s)\right]
$$

## TD3：三处设计共同控制误差放大

两个 target critic 取较小值，降低某些过高估计；target action 加小扰动进行平滑；actor 相对 critic 延迟更新。这三项不能缩写成“多训练一个网络就够了”。

$$
y=r+\gamma(1-d)\min_{j=1,2}Q_{\phi_j^-}\big(s',\mu_{\theta^-}(s')+\text{clipped noise}\big)
$$

## SAC：把随机性纳入优化目标

SAC 在奖励之外加入策略熵，actor 既追求高 Q，也保留一定随机性。下面给出现代双 Q、无单独 V 网络的常见形式；a′ 从当前策略采样，critic target 停止梯度。

$$
\begin{aligned}y&=r+\gamma(1-d)\left[\min_jQ_{\phi_j^-}(s',a')-\eta\log\pi_\theta(a'|s')\right]\\L_\pi&=\mathbb E_{s\sim D,a\sim\pi_\theta}\left[\eta\log\pi_\theta(a|s)-\min_jQ_{\phi_j}(s,a)\right]\end{aligned}
$$

连续动作通常经过 tanh 压缩到动作范围，log probability 需要相应的变量变换修正。自动调温度是另一个优化问题，不是任意把熵系数设大。

## 手算例子

r=1、γ=0.9，两个 target Q 为 4、6，η=0.2，采样动作 logπ=−0.5，且未终止。SAC target 为 1+0.9×(4−0.2×(−0.5))=4.69。若真实终止，target 就是 1。

## 对照代码

```python
# 阅读伪代码：target 与 actor 的梯度路径不同
with no_grad():
    next_action, next_logp = actor.sample(next_obs)
    soft_q = min(target_q1(next_obs, next_action), target_q2(next_obs, next_action))
    y = reward + gamma * (1 - terminated) * (soft_q - temperature * next_logp)
# actor loss 中 action 必须保留重参数化梯度；不能整段放进 no_grad
action, logp = actor.sample(obs)
actor_loss = (temperature * logp - min(q1(obs, action), q2(obs, action))).mean()
```

从 CleanRL SAC 文档进入连续动作实现，依次查重参数化采样、tanh log-prob 修正、双 Q、target 更新与温度优化。上面是机制伪代码，张量逐元素 min 需用框架的运算。

先看终止处理与动作范围；再记录真实交互次数、梯度更新次数与 buffer 容量。改成任务序列时，明确 replay、温度和归一化是否跨任务保留。

## 怎样接到 CRL

Continual World 等任务序列研究常以 SAC 为底座。更大的旧数据池不一定更好：不同任务奖励或动力学冲突时，需要明确是否提供任务信息，以及旧数据如何被使用。

适用条件：熵鼓励随机性，不保证发现所有新奖励；双 Q 也不保证不存在偏差。此处给官方深度代码入口，未将 SAC 包装成已完成的 CRL 算法。

## 思考题

SAC 的两个 Q 取 min，与 Double DQN 的“Double”是同一个公式吗？

提示：分别写出谁选动作、谁评估、最终怎样聚合。

参考答案：不是。Double DQN 分开动作选择与评估；TD3/SAC 通常对两个 critic 的估计取较小值。两者都涉及多个估计器，但运算和偏差控制方式不同。

## 原始材料与代码

- [Spinning Up · TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：按三项设计逐一理解。
- [Spinning Up · SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：先确定 soft value 的定义，再看损失。
- [CleanRL · SAC 实现](https://docs.cleanrl.dev/rl-algorithms/sac/)：从采样和 target 梯度边界开始读。
- [Continual World 官方代码](https://github.com/awarelab/continual_world)：观察 SAC 如何接到任务序列与不同 CL 方法。


---

[← 上一章](04-policy.md) · [下一章 →](06-dyna.md)
