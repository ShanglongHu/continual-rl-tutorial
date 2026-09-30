[学习首页](../README.md) · [算法与代码目录](README.md)

# 11 学什么状态：GVF、Horde、RNN 与实时递归信用

先修：TD、内部状态与参数的区别

问题：相同当前图像需要不同动作时，如何让历史成为有用状态？

观测不够 → 递归状态 / 预测状态 → GVF 与 Horde → BPTT / RTRL / RTU

一个大网络不能从没有的信息里恢复答案。首先要允许智能体把历史带到当前。其次要决定如何训练这些历史摘要：可以直接为控制学习递归表示，也可以学习对未来信号的预测。两种路线有交集，但不是同义词。

符号：zₜ：递归内部状态；θ：产生状态的参数；C：cumulant（要预测的信号，不必是任务奖励）；γₜ：延续系数；π：预测对应的策略。

## RNN 更新状态，与参数学习是两个时间过程

每步用新观测、前一动作和旧内部状态产生 z。即使 θ 暂时不变，z 也会改变；而训练 θ 决定系统将来怎样记忆。必须说明状态在 episode、task 与生命期之间何时保留。

$$
z_t=f_\theta(z_{t-1},O_t,A_{t-1}),\qquad A_t\sim\pi_\theta(\cdot\mid z_t)
$$

## GVF：用价值学习表达“将来会怎样”

把普通奖励换成感兴趣的信号，并明确预测策略与停止/折扣条件，就得到更一般的预测问题。例如继续向前走，未来电量或碰撞信号怎样累计。Horde 将许多这样的预测学习器组织在同一经验流上。

$$
v(s)=\mathbb E_\pi\left[\sum_{k\ge0}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}\mid S_t=s\right]
$$

k=0 的空乘积为 1。若行为策略不是预测策略，不能直接假设普通 on-policy TD 安全；需要考虑 off-policy 学习算法与条件。

## 历史梯度怎样到达参数：BPTT 与 RTRL

BPTT 保存或重建一段计算图向后传播，截断会丢掉窗口外的梯度路径。RTRL 向前维护状态对参数的敏感度，但一般稠密递归网络成本很高。RTU 等利用结构约束降低成本，不是任意 RNN 的免费精确梯度。

$$
H_t=\frac{\partial z_t}{\partial\theta}=\frac{\partial f_\theta}{\partial z_{t-1}}H_{t-1}+\frac{\partial f_\theta}{\partial\theta}
$$

这里沿给定输入序列求递归导数；实际 RL 的数据生成与策略梯度还有额外问题。

## 手算例子

进门看到红灯，随后十步走廊完全相同，最后需要选左。手工保存灯色即可解决信息问题；如果 RNN 没学到这点，可能是训练目标或长程信用分配失败，而不是任务本身不可解。

## 对照代码

```python
# 状态与参数的生命周期要分别追踪
hidden = recurrent_cell(obs, previous_action, hidden)
prediction = prediction_head(hidden)        # 可预测 cumulant
action = policy_head(hidden).sample()
# detach(hidden) 截断梯度历史，不会自动把 hidden 的数值清零
# reset hidden / optimizer / replay 是三个不同操作
```

先运行现有 memory 诊断确认信息条件；再读 RTU 作者仓库中的递归单元和更新循环。Horde 原论文说明预测问题与学习器组织，不能用一段 RNN forward 代码替代它。

```sh
python3 examples/crl_labs.py memory --seeds 1 --steps 1000 --out memory-study.csv
```

手工 oracle 应达到 1，无记忆固定策略在平衡线索中为 0.5。然后才研究“学习出的记忆”，并匹配历史、状态维度和每步计算；不能把 oracle 当成训练 RNN 的结果。

## 怎样接到 CRL

连接 Forager、POPGym、预测式 agent state 与实时递归学习。持续学习中，不仅世界变化，策略变化也会改变需要记住和预测的对象。

适用条件：多个预测准确，不保证它们足够支持控制；工作记忆丰富，也不自动代表长期知识持续增长。先为你的主张选择对应的测试。

## 思考题

对 hidden 做 detach 与重置为零，对行为和梯度有什么不同？

提示：区分数值和计算图。

参考答案：detach 保留当前数值、切断跨边界梯度；置零改变内部信息，可能立刻改变行为。两者都影响学习实验，但不是同一权限。

## 原始材料与代码

- [Horde 原论文](https://josephmodayil.com/papers/horde-final.pdf)：先看一般预测问题，再看多学习器架构。
- [RTU 原论文](https://arxiv.org/abs/2409.01449)：追实时梯度的结构假设。
- [RTU 作者代码](https://github.com/esraaelelimy/rtus)：定位递归状态与梯度计算。


---

[← 上一章](10-streaming.md) · [下一章 →](12-meta.md)
