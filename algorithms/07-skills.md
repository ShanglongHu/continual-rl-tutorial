[学习首页](../README.md) · [算法与代码目录](README.md)

# 07 积累可复用知识：Options、SR、Successor Features 与 GPI

先修：Bellman、Dyna、向量内积；先看入门第 6 课

问题：换了奖励以后，哪些行为结构可以不从头学习？

Options：行为跨多步 → 技能模型：预测后果 → SR / SF：未来占用 → GPI：复用多种策略

这里有两条相交但不同的线：options 把多步行为变成决策单位；SR/SF 把环境中的未来占用与奖励权重分开。Machado 的技能发现工作会利用表示来构造探索技能，但不能把 option、SR 和规划混为一个对象。

符号：o=(I,π,β)：启动集合、内部策略和终止函数；τ：技能持续时间；φ：转移特征；w：奖励权重；ψπ：策略 π 下的折扣特征累计。

## Option 的后续价值按真实持续时间折扣

执行一个技能会跨越 τ 个基本时间步。把这段内部奖励先累计，再在技能结束后选下一个可用技能。Q 在这里指给定技能集合上的最优技能价值。

$$
Q(s,o)=\mathbb E\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'}Q(S_{t+\tau},o')\mid S_t=s,o\right]
$$

## SR 与 SF：先预测会遇到什么，再赋予价值

SR 预测状态的折扣访问次数；SF 将它推广为特征的折扣累计。若即时奖励能写为 φᵀw，就可将价值分解为未来特征与奖励权重的内积。

$$
r(s,a,s')=\phi(s,a,s')^\top w,\quad \psi^\pi(s,a)=\mathbb E_\pi\left[\sum_{k\ge0}\gamma^k\phi_{t+k+1}\right],\quad Q_w^\pi(s,a)=\psi^\pi(s,a)^\top w
$$

## GPI：在已有策略的预测中选择动作

保存若干策略的 successor features 后，对新奖励 w 重新评估它们。GPI 对每个动作取已有策略预测中的最好值，再选动作。改进保证需要任务共享结构与估计精度等条件，不能用一个 max 省略这些假设。

$$
\pi_{\mathrm{GPI}}(s)\in\arg\max_a\max_i\left[\psi^{\pi_i}(s,a)^\top w\right]
$$

## 手算例子

未来特征的预测为 ψ=(2,1)。旧奖励权重 w=(1,0)，价值为 2；新权重 (0,3)，价值变为 3，不必为这个固定策略重新采完整回报。但若道路结构也改变，ψ 本身可能已经失准。

## 对照代码

```python
# SF 的 on-policy 一步更新：向量 TD
target = phi if terminal else phi + gamma * psi[next_state, next_action]
psi[state, action] += alpha * (target - psi[state, action])
q_for_new_reward = psi[state, action] @ new_reward_weights
# Options 还需要 initiation / intra-option policy / termination 与技能模型
```

先读 Machado options 仓库的 main.py 和 README，识别环境、option 构造与策略执行；再对照课程 slides 中的表示驱动技能发现。SF 片段是本页的机制示意，不声称该仓库实现了这里全部 SF/GPI 流程。

先在小网格中只改变奖励，再改变动力学。比较“只改 w”“更新 ψ”“重新学习”三种条件；否则不知道迁移失败在哪个分解假设。

## 怎样接到 CRL

这是从“适应变化”走向“知识积累”的重要一支：不仅防止旧能力丢失，还要让既有结构帮助未来任务。但技能数量增加不等于规划收益增加。

适用条件：经典 SF 的便捷重估依赖奖励表示与共享动力学。Option 的发现、执行、模型学习和规划应分别测量；未实现的环节不由表示图自动补齐。

## 思考题

世界动力学不变，但用于计算 ψ 的策略已经更新，旧 ψ 必然仍准确吗？

提示：ψ 上标是 π。

参考答案：不必然。它预测的是特定策略下的未来特征分布；策略变化也可能使它过时。可以保留旧策略及其 ψ，或跟踪更新新的 ψ。

## 原始材料与代码

- [SF 与 GPI 原论文](https://arxiv.org/abs/1606.05312)：抓住奖励分解及共享环境假设。
- [Machado 课程与 slides](https://deeprlcourse.github.io/guests/marlos_machado/)：先读 option 定义，再读表示如何用于发现技能。
- [Machado options 代码](https://github.com/mcmachado/options)：从 main.py 找到运行入口，按 README 建独立环境。


---

[← 上一章](06-dyna.md) · [下一章 →](08-retention.md)
