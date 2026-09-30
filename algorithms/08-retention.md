[学习首页](../README.md) · [算法与代码目录](README.md)

# 08 怎样保留旧知识：Replay、EWC、蒸馏与参数隔离

先修：梯度下降；区分旧任务评测与新任务学习

问题：为什么一个保护旧任务的方法，会让新任务更难学？

Fine-tuning 基线 → Replay：保留数据 → EWC：限制参数 → 蒸馏 / 参数隔离

先明确旧知识仍然值得保留。若只是同一任务的旧奖励规律已经失效，强行保留它可能妨碍适应。若未来还会回访旧任务，保留才是明确目标。不同方法保护的对象不同，需要对应的 memory、task ID 与计算权限。

符号：L_B：新任务损失；θ_A：学完旧任务的参数；Fᵢ：旧任务参数重要性的对角近似；κ：保护强度；β：旧数据损失权重。

## Replay：让旧样本继续参与优化

概念上可写成新旧数据损失的加权和。RL 里旧 transition 来自过去策略；如何构造当前 target、做 off-policy 校正或处理过时奖励，需要看具体底座，不能直接照搬监督混合数据。

$$
L(\theta)=(1-\beta)\mathbb E_{D_B}\ell(\theta)+\beta\mathbb E_{D_A}\ell(\theta)
$$

## EWC：对重要参数加弹簧

把旧任务解附近的损失/后验近似成二次形式，保留中心 θ_A 和重要性。常见 EWC 用对角 Fisher 近似；这是局部近似，不是所有参数移动都等价于遗忘。

$$
L(\theta)=L_B(\theta)+\frac\kappa2\sum_iF_i(\theta_i-\theta_{A,i})^2,\quad \nabla_iL=\nabla_iL_B+\kappa F_i(\theta_i-\theta_{A,i})
$$

## 蒸馏与参数隔离保护的是另外两件事

蒸馏要求新网络在选定输入上保留旧输出；没覆盖到的状态仍可能丢失。Progressive networks 或 PackNet 一类方法保留/划分参数，换来容量增长、掩码或任务路由等成本。Progress & Compress 则把快速适应与知识整合分开。它们不应只用一个最终分数比较。

## 手算例子

标量旧任务希望 w=+1，新任务希望 w=−1，F=1。EWC 目标为 ½(w+1)²+κ/2(w−1)²，最优解 w*=(κ−1)/(κ+1)。κ=0 得到 −1；κ=1 得到 0；κ 越大越靠近旧解，也越难满足新目标。

## 对照代码

```python
# consolidation() 中的已知曲率教学例子，不是神经网络 Fisher 估计
gradient = w + 1                           # 新目标 L_B 的梯度
gradient += strength * (w - 1)             # F=1，旧解为 +1
w -= 0.05 * gradient
# equal_rehearsal 则用 0.5*(w+1) + 0.5*(w-1)
```

先跑标量例子看 trade-off，再读 Continual World 的 CL 方法实现，或 AGI-Labs continual_rl 的策略接口、任务序列与评测。不要把“允许访问旧数据”和“不给旧数据”放在一张无协议说明的表里。

```sh
python3 examples/rl_foundations.py consolidation --seeds 1 --strength 1 --out retention-run
```

分别画 old_loss 和 new_loss；把 --strength 改为 0、1、5，检查稳态是否接近解析解。它是确定性例子，不用多 seed 制造重复证据。

## 怎样接到 CRL

CRL 中不仅有遗忘，也有合理丢弃过时信息的需求。研究一个 retention 方法前，先决定旧任务何时会回来、是否可评估，以及 task ID 是否可用。

适用条件：这个标量实验是监督二次损失类比，不是 EWC 的完整 RL 复现。现实网络的 Fisher 估计、任务边界和多任务重要性累积都需要额外实现。

## 思考题

在 κ=1 的例子里，EWC 与等权 rehearsal 最优解相同，学习曲线也必然相同吗？

提示：比较两者的梯度幅度，而不只看最优点。

参考答案：不必然。EWC 梯度为 2w，等权 rehearsal 为 w。相同步长下更新速度不同；目标的整体缩放与学习率也会影响比较。

## 原始材料与代码

- [EWC 原论文](https://arxiv.org/abs/1612.00796)：读参数重要性和旧任务附近的近似。
- [Progress & Compress](https://arxiv.org/abs/1805.06370)：理解快速适应与慢速整合的分工。
- [Continual World](https://github.com/awarelab/continual_world)：查看 task sequence、CL 方法与 SAC 底座。
- [Continual RL 基线库](https://github.com/AGI-Labs/continual_rl)：追策略接口、实验定义与公共指标。


---

[← 上一章](07-skills.md) · [下一章 →](09-plasticity.md)
