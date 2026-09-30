[学习首页](../README.md) · [算法与代码目录](README.md)

# 12 让未来学得更快：迁移、Meta-RL 与自动课程

先修：策略梯度、任务分布、训练与评测分离

问题：一次新任务学得快，是否意味着能够终身积累？

Transfer：复用已有知识 → MAML：可适应初始化 → RL²：递归学习规则 → Curriculum：选择经验

这几条路线都与未来学习有关，但优化对象不同。迁移复用表示、价值或技能；meta-learning 优化跨任务适应过程；课程学习选择接下来遇到的任务。将三者统称为“自动学习”会掩盖实际假设。

符号：θ：共享初始化；Lᵢᴬ：任务 i 的适应损失；LᵢQ：适应后的评估损失；α：内部步长；p(i)：任务分布。RL 中常把负回报或策略代理目标写成损失。

## MAML：优化经过学习后的表现

先在任务 i 上用适应数据走一步，再用分离的评估数据衡量更新后的参数。外层不是只优化当前 θ 的分数，而是优化“从 θ 开始学一次”的结果。

$$
\theta_i'=\theta-\alpha\nabla_\theta L_i^A(\theta),\qquad\min_\theta\;\mathbb E_{i\sim p(i)}L_i^Q(\theta_i')
$$

## 为什么会出现二阶项

更新后的 θ′ 也依赖 θ，对外层目标应用链式法则，就有内部更新映射的导数。一阶 MAML 近似忽略这一 Hessian 项；它是近似，而不是完全相同的目标梯度。

$$
\nabla_\theta L_i^Q(\theta_i')=\left(I-\alpha\nabla_\theta^2L_i^A(\theta)\right)^\top\nabla_{\theta_i'}L_i^Q(\theta_i')
$$

## RL² 与自动课程分别改变什么

RL² 用跨交互保留的递归状态实现任务内适应，慢速外层 RL 训练其行为规则。自动课程则选择任务、目标或环境参数，让学习器获得更有用的经验。学习进展估计要区别于原始预测误差：不可约噪声很大，也可能几乎没有可学进展。

## 与终身 CRL 的接口

Meta-test 常从同一个初始化重启，CRL 则可能要求保留整段生命史；课程 teacher 也可能知道任务 ID、真实难度或成功标签。先列权限，再讨论迁移收益和长期知识积累。

## 手算例子

两个任务各自适应后都很好，但每次都从 θ 重启，这说明初始化便于适应；不能据此声称一位 agent 连续经历任务后越学越强。若每个任务都重置 hidden state，也要说明 RL² 允许记忆跨越哪些 episode。

## 对照代码

```python
# MAML 机制伪代码；support 与 query 数据必须分开
adapted = theta - inner_lr * grad(adaptation_loss(task, theta))
meta_loss = evaluation_loss(task, adapted)
meta_gradient = differentiate_through_update(meta_loss, theta)
# RL²: 重点检查 recurrent state 跨 episode 的保留与 task 边界
# Curriculum: 重点检查 teacher 选择任务时能看到什么指标
```

RL² 可用 garage 文档定位训练循环、任务采样与 recurrent policy；它是第三方参考实现。TeachMyAgent 提供课程算法与测试环境入口。分别读 teacher 和 student，不把环境生成器当成 agent。

划分训练任务与未见评测任务；匹配适应交互预算；增加无迁移初始化、随机课程等基线。另画不重启学习器的持续任务流，才能讨论它是否帮助 CRL。

## 怎样接到 CRL

与技能迁移、开放式课程、多智能体对手变化相接。下一步可沿研究地图的元学习/课程分支选一个协议，而不是把全部机制一次塞入同一 agent。

适用条件：MAML 的经典任务分布假设、RL² 的记忆协议和开放式环境生成不是同一个问题。这里的二阶公式先在可微损失下解释；完整 RL meta-gradient 还涉及采样分布。

## 思考题

课程 teacher 总挑预测误差最大的任务，为什么可能一直挑随机噪声？

提示：误差很大与误差正在下降是不同量。

参考答案：不可约噪声可以持续产生大误差，但训练未必带来进步。应衡量可学习的进展、任务多样性和覆盖，并与随机课程比较。

## 原始材料与代码

- [MAML 原论文](https://arxiv.org/abs/1703.03400)：读“适应之后”的外层目标。
- [RL² 原论文](https://arxiv.org/abs/1611.02779)：关注慢速参数学习与快速递归适应。
- [garage · RL² 实现说明](https://garage.readthedocs.io/en/latest/user/algo_rl2.html)：第三方实现，先按其版本文档建立环境。
- [Narvekar 等课程学习综述](https://www.jmlr.org/papers/v21/20-212.html)：区分任务生成、排序和知识迁移。
- [TeachMyAgent 官方代码](https://github.com/flowersteam/TeachMyAgent)：找到 teacher/student 边界与基线。


---

[← 上一章](11-state.md) · [下一章 →](13-exploration.md)
