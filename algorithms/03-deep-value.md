[学习首页](../README.md) · [算法与代码目录](README.md)

# 03 当 Q 表放不下：DQN、Double DQN 与数据回放

先修：Q-learning；梯度下降与神经网络

问题：把表格换成网络，为什么还需要 target network 和 replay？

Q-learning → 共享参数 Qθ → DQN → Double DQN / 回报分布

表格更新只动一个格子；网络的一次更新可能同时改变许多状态动作的预测。再加上 target 依赖同一个网络、策略不断改变数据分布，训练不再像普通固定数据回归。DQN 的训练机制正是在处理这些耦合。

符号：θ：在线网络；θ⁻：滞后的 target 网络；d：真正终止标记；D：replay buffer；sg：停止梯度。

## 先看损失，再看两个稳定化设计

从 replay 采一批 transition，用滞后网络构造目标，再只更新在线网络。Replay 改变采样相关性并重复利用数据；target network 让目标短期内变化较慢。两者不是同一机制，也不构成一般稳定性证明。

$$
y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a'),\qquad L(\theta)=\mathbb E_D\left[\tfrac12\big(Q_\theta(s,a)-\operatorname{sg}(y)\big)^2\right]
$$

## Double DQN：分开选动作与评动作

对带噪声的估计取 max，可能偏向恰好被高估的动作。Double DQN 用在线网络选动作，再用 target 网络评估该动作。它不是“两个 Q 取最小”，后者常见于 TD3/SAC。

$$
a^*=\arg\max_a Q_\theta(s',a),\qquad y_{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s',a^*)
$$

## 往外扩展时，分清改动对象

Prioritized replay 改采样分布，通常需要重要性权重控制偏差；dueling 改网络分解；C51 / QR-DQN 学回报分布。它们可以组合，但“预测回报分布”不等于已经识别环境变化或解决遗忘。

## 手算例子

下一状态在线网络 Q 为 (5,4)，target 网络为 (3,6)，r=0、γ=0.9。DQN target 是 5.4；Double DQN 先选在线网络偏好的第一个动作，再得到 2.7。两个更新公式的差异可以在一个数值例子里看到。

## 对照代码

```python
# 阅读伪代码；对应官方实现的 target / loss / optimizer 部分
with no_grad():
    next_action = online(next_obs).argmax(dim=1)
    next_q = target(next_obs).gather(1, next_action[:, None])
    y = reward + gamma * (1 - terminated) * next_q
loss = 0.5 * ((online(obs).gather(1, action) - y) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

从 CleanRL 的 DQN 文档进入 dqn.py。先标记 replay 写入/采样、target 同步、终止处理，再看优化器。上面的片段是 Double DQN 的机制示意；不要把它当成 CleanRL dqn.py 的逐字摘录。

先在官方原任务跑通，再单独加入奖励变化。记录 buffer 中新旧数据比例；比较不同容量时要匹配梯度更新预算，而不仅是环境步数。

## 怎样接到 CRL

在 CRL 中，旧数据可能保护旧任务，也可能过时；replay 容量和采样策略因此成为研究对象。DQN 的回放不是为长期保留而自动设计好的。

适用条件：本章提供公式与官方深度实现入口；标准库教学包不实现 DQN。off-policy、函数逼近与自举的组合需要额外稳定性分析。

## 思考题

把 batch size 从 64 改成 1，就变成 streaming RL 了吗？

提示：样本来自新 transition 还是 replay？是否仍有 target network？

参考答案：不一定。若依旧从旧 buffer 采样，仍在回放。应报告数据重用、目标网络、更新频率与每步计算，而不是只报 batch size。

## 原始材料与代码

- [CleanRL DQN 文档与代码](https://docs.cleanrl.dev/rl-algorithms/dqn/)：先追一条 transition 的生命周期。
- [Double DQN 原论文](https://arxiv.org/abs/1509.06461)：对照动作选择与动作评估分离。
- [Spinning Up 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：定位价值学习与策略优化两条路线。


---

[← 上一章](02-control.md) · [下一章 →](04-policy.md)
