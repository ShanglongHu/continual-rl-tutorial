# 强化学习基础：表格方法、函数逼近与深度学习

每章给出设定、推导、执行顺序、手算、代码、边界与练习。Part II 不是可跳过的附录：共享参数、半梯度、离策略稳定性、资格迹和策略梯度是后续方法的共同基础。

## [表格强化学习 · Sutton Part I](tabular/README.md)

- [第 1 章 · 多臂老虎机：估计、探索与直接策略学习](tabular/bandits.md)
- [第 2 章 · MDP、回报与价值：序列决策的数学对象](tabular/mdps.md)
- [第 3 章 · 动态规划：评价、改善与最优递推](tabular/dynamic-programming.md)
- [第 4 章 · Monte Carlo：完整回报、探索控制与离策略评价](tabular/monte-carlo.md)
- [第 5 章 · TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](tabular/temporal-difference.md)
- [第 6 章 · 多步学习：n-step、Tree Backup 与 Q(σ)](tabular/multistep.md)
- [第 7 章 · 学习与规划：Dyna、优先扫描和执行时搜索](tabular/planning.md)

## [函数逼近与经典进阶 · Sutton Part II](approximation/README.md)

- [第 1 章 · 函数逼近预测：从回归到 TD 固定点](approximation/prediction.md)
- [第 2 章 · 特征、泛化与半梯度控制](approximation/features-control.md)
- [第 3 章 · 持续控制与平均奖励](approximation/average-control.md)
- [第 4 章 · 离策略函数逼近：覆盖、发散与稳定更新](approximation/off-policy.md)
- [第 5 章 · 多步回报、资格迹与 True-online TD](approximation/traces.md)
- [第 6 章 · 策略梯度、基线与 Actor–Critic](approximation/policy-gradient.md)

## [现代深度强化学习 · 核心算法与并列研究分支](deep/README.md)

- [第 1 章 · 深度价值学习：DQN、Double DQN 与目标的时间顺序](deep/deep-value.md)
- [第 2 章 · 策略梯度：从轨迹概率到 GAE 与 actor–critic](deep/policy-gradient.md)
- [第 3 章 · 策略更新的尺度：TRPO 与 PPO](deep/trust-region.md)
- [第 4 章 · 连续动作的价值优化：DDPG 与 TD3](deep/deterministic-control.md)
- [第 5 章 · 最大熵连续控制：SAC 的价值、密度与温度](deep/entropy-control.md)
- [第 6 章 · 深度 RL 的机制接口：模型、记忆、离线数据与实验](deep/practice.md)
- [第 7 章 · 大规模训练：算法与系统怎样共同设计](deep/systems.md)
- [不完全可观测：信念状态、信息行动与递归记忆](deep/partial-observability.md)
- [探索与不确定性：后验、乐观估计和时间一致行动](deep/exploration.md)
- [分布强化学习：Bellman 分布、分位数与风险目标](deep/distributional.md)
- [离线强化学习：数据支持、策略评估与保守改进](deep/offline.md)
- [模型学习与规划：MPC、短模型 rollout 和潜在想象](deep/model-based.md)
- [约束强化学习：占据测度、拉格朗日与可行策略](deep/constraints.md)
- [多智能体合作：结构化探索与信用分配](deep/multi-agent.md)
- [自对弈与开放式多智能体学习：评估、目标与策略种群](deep/multi-agent-populations.md)
- [对手建模与递归推理：预测谁，回应什么？](deep/multi-agent-reasoning.md)

[CRL 教材](../textbook/README.md)
