[学习首页](../README.md) · [算法与代码目录](README.md)

# 09 怎样一直学得进：ReDo、Continual Backprop 与特征更新

先修：梯度与神经网络；入门第 3 课的 old/fresh 诊断

问题：旧任务没忘，不代表还有能力学新内容；这种能力怎样维持？

先诊断学习能力 → ReDo：低活跃单元 → CBP：低效用特征 → 长期对照与知识积累

Retention 问“以前会的还会不会”，plasticity 问“接下来还学不学得进”。它们可能冲突，也可能同时受损。重置特征的一类方法尝试保留有用知识，同时为新学习留出可调整的表示。首先要分清机制信号与任务表现。

符号：hᵢ(x)：单元激活；H：该层单元数；sᵢ：归一化活跃度；uᵢ：效用统计；ρ：效用衰减系数。效用定义随算法版本而异。

## ReDo 先识别相对低活跃的单元

将一个单元平均绝对激活除以层平均，可以比较同一层中的相对活跃程度。达到低活跃阈值后，重置输入连接、控制输出连接，使新特征重新有机会参与。分母接近零时，实际代码还需要数值处理。

$$
s_i=\frac{\mathbb E_x|h_i(x)|}{H^{-1}\sum_{j=1}^H\mathbb E_x|h_j(x)|}
$$

## Continual Backprop：普通学习之外持续生成与筛选

除了 backprop 更新，还持续估计特征效用；只在达到一定年龄的特征中替换少量低效用单元，避免新单元还没学就被淘汰。效用可以涉及激活及输出贡献，不能把它直接等同于 ReDo 的低活跃度。

$$
u_{i,t}=\rho u_{i,t-1}+(1-\rho)c_{i,t}
$$

这只是指数平均的记账形式；cᵢ 的具体定义、偏差修正、年龄与替换预算必须回到对应版本论文/实现。不是用这一式就复现了 CBP。

## 重置一个神经元，也涉及优化器状态

需要一起检查输入权重、输出权重、偏置、Adam 等优化器统计与替换顺序。若只改权重而沿用不匹配的旧动量，后续更新可能立即改变新特征。还要测原任务是否被破坏，而不是只数有多少单元重新活跃。

## 手算例子

一个单元的平均绝对激活为 0.01，所在层平均为 0.5，相对活跃度为 0.02。这说明它在这批输入上低活跃，不说明它在所有稀有状态中无用。替换后的新任务收益和旧任务损失必须另外测量。

## 对照代码

```python
# 机制伪代码；不是任一论文全部实现
normal_gradient_update(batch)
update_feature_statistics(activations, outgoing_weights)
eligible = units_with_sufficient_age()
chosen = select_low_utility(eligible, replacement_budget)
reinitialize_incoming_weights(chosen)
reset_outgoing_weights_and_optimizer_state(chosen)
```

作者 loss-of-plasticity 仓库：lop/algos 看学习与特征更新，lop/nets 看网络，lop/slowly_changing_regression 看小任务，再进 lop/rl。先核对具体版本的效用定义。ReDo 从论文及其代码入口比较，不把两种选择规则混写。

最小对照包括不替换、匹配频率的随机替换、效用替换；另设 fresh reference。匹配网络、数据和学习率调参预算，测新任务曲线、旧任务保留及替换成本。

## 怎样接到 CRL

它与 streaming 可组合，但一次重置并不解决在线信用分配，也不自动形成知识积累。前沿问题是何时替换、替换什么，以及如何避免持续探索与保护旧知识互相干扰。

适用条件：低活跃、低秩、梯度变小都是诊断线索，不能代替可学习性指标。这里提供作者研究代码，不用一个表格/线性小实验声称复现深度可塑性。

## 思考题

新方法让 dormant neuron 比例下降，回报却不变，应怎样总结？

提示：测量量与科学主张是否对应？

参考答案：可以说改变了活跃度统计；还不能说改善了控制或长期学习能力。检查新任务学习、旧知识和不同任务分布，保留没有收益的结果。

## 原始材料与代码

- [ReDo · ICML 原论文与入口](https://proceedings.mlr.press/v202/sokar23a.html)：重点看低活跃度定义及替换过程。
- [Loss of Plasticity · Nature](https://www.nature.com/articles/s41586-024-07711-7)：从实验设置和新学习曲线理解现象。
- [Continual Backprop 作者代码](https://github.com/shibhansh/loss-of-plasticity)：按 algos → 小回归任务 → RL 阅读。


---

[← 上一章](08-retention.md) · [下一章 →](10-streaming.md)
