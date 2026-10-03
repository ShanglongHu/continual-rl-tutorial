# 实验与实现

在仓库根执行。标准库机制脚本无需额外安装。可选神经训练需要 PyTorch，安装与测试见对应章节。单元测试、短程训练和原论文效能复现是不同验收层。

```bash
python3 scripts/test_examples.py
python3 scripts/run_all.py --quick --out results/first-run
```

[实验设计与测试手册](experiment-handbook.md) · [研究协议](protocol.md) · [研究 Workbench](https://github.com/ying-wen/rl-research-workbench)

## [多臂老虎机：估计、探索与直接策略学习](../foundations/tabular/bandits.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py bandits
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [MDP、回报与价值：序列决策的数学对象](../foundations/tabular/mdps.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py mdps
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [动态规划：评价、改善与最优递推](../foundations/tabular/dynamic-programming.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py dynamic-programming
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [Monte Carlo：完整经历、重复访问与离策略评价](../foundations/tabular/monte-carlo.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py monte-carlo
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](../foundations/tabular/temporal-difference.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py temporal-difference
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [多步学习：n-step、Tree Backup 与 Q(σ)](../foundations/tabular/multistep.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py multistep
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [学习与规划：Dyna、优先扫描和执行时搜索](../foundations/tabular/planning.md)

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

```bash
python3 examples/tabular_textbook_lab.py planning
python3 examples/tabular_textbook_lab.py test
```

[源码](../examples/tabular_textbook_lab.py)

## [函数逼近预测：从回归到 TD 固定点](../foundations/approximation/prediction.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py prediction
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [特征、泛化与半梯度控制](../foundations/approximation/features-control.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py features-control
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [持续控制与平均奖励](../foundations/approximation/average-control.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py average-control
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [离策略函数逼近：覆盖、发散与稳定更新](../foundations/approximation/off-policy.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py off-policy
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [多步回报、资格迹与 True-online TD](../foundations/approximation/traces.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py traces
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [策略梯度、基线与 Actor–Critic](../foundations/approximation/policy-gradient.md)

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

```bash
python3 examples/approximation_textbook_lab.py policy-gradient
python3 examples/approximation_textbook_lab.py test
```

[源码](../examples/approximation_textbook_lab.py)

## [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)

CPU DeadlineChain 完整训练；需要同目录 deep_textbook_lab.py 和 PyTorch。

```bash
python3 examples/deep_textbook_train.py dqn --steps 2000 --seed 0
```

[源码](../examples/deep_textbook_train.py)

## [策略梯度：从轨迹概率到 GAE 与 actor–critic](../foundations/deep/policy-gradient.md)

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

```bash
python3 examples/deep_textbook_lab.py test
```

[源码](../examples/deep_textbook_lab.py)

## [策略更新的尺度：TRPO 与 PPO](../foundations/deep/trust-region.md)

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

```bash
python3 examples/deep_textbook_lab.py test
```

[源码](../examples/deep_textbook_lab.py)

## [连续动作的价值优化：DDPG 与 TD3](../foundations/deep/deterministic-control.md)

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

```bash
python3 examples/deep_textbook_lab.py test
```

[源码](../examples/deep_textbook_lab.py)

## [最大熵连续控制：SAC 的价值、密度与温度](../foundations/deep/entropy-control.md)

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

```bash
python3 examples/deep_textbook_lab.py test
```

[源码](../examples/deep_textbook_lab.py)

## [深度 RL 的机制接口：模型、记忆、离线数据与实验](../foundations/deep/practice.md)

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

```bash
python3 examples/deep_textbook_lab.py test
```

[源码](../examples/deep_textbook_lab.py)

## [任务与优化目标：奖励、回报和持续交互](../textbook/objectives.md)

标准库解析实验：周期策略偏好、随机停止、终止与截断、势函数塑形、在线与冻结评价。

```bash
python3 examples/objectives_lab.py all
```

[源码](../examples/objectives_lab.py)

## [平均奖励：奖励率、差分价值与持续控制](../textbook/average.md)

标准库表格机制实验与单步公式测试；不是深度算法或论文 benchmark 复现。

```bash
python examples/lifelong_algorithms_lab.py average
```

[源码](../examples/lifelong_algorithms_lab.py)

## [Agent state：部分可观测性、递归记忆与在线信用分配](../textbook/state.md)

Bayes 过滤、标量 RTRL/BPTT/TBPTT、重置式监督提示实验、单个线性 RTU 风格旋转块；不是原论文 RL 基准复现。

```bash
python3 examples/state_meta_lab.py state
```

[源码](../examples/state_meta_lab.py)

## [价值预测与时间差分学习](../textbook/value.md)

Python 3.10+，仅标准库。实现表格 MC/TD 完整小实验；资格迹的前后向推导与实现见时间信用分配章。

```bash
python3 examples/foundations_detail_lab.py value
python3 examples/foundations_detail_lab.py test
```

[源码](../examples/foundations_detail_lab.py)

## [通用价值函数与预测知识](../textbook/gvf.md)

六个线性学习器、三个共享经验的预测问题、解析 Bellman 参照与一个 off-policy 期望发散反例。

```bash
python3 examples/gvf_lab.py compare
python3 examples/gvf_lab.py test
```

[源码](../examples/gvf_lab.py)

## [控制问题：策略改进与动态规划](../textbook/control.md)

Python 3.10+，仅标准库。包含策略迭代、价值迭代、完整在线 Q-learning 与 12 个测试。确定性两状态教学环境；不作为复杂任务的性能证据。

```bash
python3 examples/control_problem_lab.py all
python3 examples/control_problem_lab.py test
```

[源码](../examples/control_problem_lab.py)

## [深度价值学习：DQN 与 Double DQN](../textbook/deep-value.md)

Python 3.10+，仅标准库。包含两层网络及手写反向传播、replay 与目标网络。实验环境为可解析的小型 MDP。

```bash
python3 examples/foundations_detail_lab.py deep-value
python3 examples/foundations_detail_lab.py test
```

[源码](../examples/foundations_detail_lab.py)

## [策略梯度、Actor–Critic 与 PPO](../textbook/policy.md)

Python 3.10+，仅标准库。执行 PPO bandit 完整采样/优化循环，独立实现多步 GAE；不声称包含网络 PPO 的通用训练工程。

```bash
python3 examples/foundations_detail_lab.py policy
python3 examples/foundations_detail_lab.py test
```

[源码](../examples/foundations_detail_lab.py)

## [最大熵控制与 Soft Actor–Critic](../textbook/soft-control.md)

Python 3.10+，仅标准库。运行精确离散 actor 优化和公式检查；连续 SAC 的完整训练顺序在正文，作者工程在章末。

```bash
python3 examples/foundations_detail_lab.py soft-control
python3 examples/foundations_detail_lab.py test
```

[源码](../examples/foundations_detail_lab.py)

## [时间信用分配：多步回报、资格迹与在线等价](../textbook/credit.md)

标准库：冻结前后向等价、传统与 true-online TD、在线前向参考、SARSA／Watkins 时序、最小递归敏感度。

```bash
python3 examples/credit_assignment_lab.py all
```

[源码](../examples/credit_assignment_lab.py)

## [流式强化学习：交互协议与更新稳定性](../textbook/streaming.md)

流式 TD、在线统计与 ObGD 的机制检查；meta_frontier_lab.py 提供 Intentional 输出尺度实验。不包含完整神经网络控制训练。

```bash
python examples/lifelong_algorithms_lab.py streaming
```

[源码](../examples/lifelong_algorithms_lab.py)

## [学习规则的适应：在线元梯度与跨任务元学习](../textbook/meta.md)

多步超梯度、IDBD/TIDBD、标量 MAML 与解析 context；meta_frontier_lab.py 提供 Metatrace 线性特例。

```bash
python3 examples/state_meta_lab.py meta
```

[源码](../examples/state_meta_lab.py)

## [目标与子任务：条件控制、经验重用与技能设计](../textbook/goals.md)

目标重标记、子任务 TD 与有限目标课程的表格实验；深度方法另附原论文和作者工程。

```bash
python3 examples/knowledge_algorithms_lab.py goals
```

[源码](../examples/knowledge_algorithms_lab.py)

## [Options：多步决策、技能发现与可复用行为](../textbook/options.md)

SMDP、intra-option、动作与终止梯度的表格实验；深度技能训练另附原论文和作者工程。

```bash
python3 examples/knowledge_algorithms_lab.py options
```

[源码](../examples/knowledge_algorithms_lab.py)

## [Dyna：模型学习与规划](../textbook/dyna.md)

Python 3.10+，仅标准库。完整表格 Dyna-Q 与解析参照；概率模型、优先队列与神经世界模型需按正文接口进一步实现。

```bash
python3 examples/foundations_detail_lab.py dyna
python3 examples/foundations_detail_lab.py test
```

[源码](../examples/foundations_detail_lab.py)

## [模型与后果预测：学什么，才能用于下一次决策？](../textbook/models.md)

表格 option model、SF/GPI 与条件反例；不包含完整深度世界模型训练。

```bash
python3 examples/knowledge_algorithms_lab.py models
```

[源码](../examples/knowledge_algorithms_lab.py)

## [规划：把模型中的经验转成更好的决策](../textbook/planning.md)

表格 Dyna、优先传播、option value iteration 与枚举 MPC；深度规划和想象训练另附原论文及作者工程。

```bash
python3 examples/knowledge_algorithms_lab.py planning
```

[源码](../examples/knowledge_algorithms_lab.py)

## [知识保留：经验重放、参数约束与模型记忆](../textbook/retention.md)

存储策略、目标项和 trace 数值实验；完整 CLEAR 训练还需要神经网络、序列收集与环境配置。

```bash
python examples/lifelong_algorithms_lab.py retention
```

[源码](../examples/lifelong_algorithms_lab.py)

## [可塑性：梯度通路、有效学习率与预测干扰](../textbook/plasticity.md)

替换、optimizer 状态与合成 aged/fresh 机制诊断；不是 ReDo/CBP 全论文复现。

```bash
python examples/lifelong_algorithms_lab.py plasticity
```

[源码](../examples/lifelong_algorithms_lab.py)

## [持续探索：新奇、不确定性、学习进展与恢复](../textbook/exploration.md)

内部奖励/进展/恢复接口的确定性检查；非 RND、课程或安全算法完整性能复现。

```bash
python examples/lifelong_algorithms_lab.py exploration
```

[源码](../examples/lifelong_algorithms_lab.py)

## [持续智能体架构：模块接口、更新调度与长期评价](../textbook/architectures.md)

六状态、双动作、手工记忆的集成教学骨架；不是完整 OaK/STOMP 性能复现。

```bash
python examples/lifelong_algorithms_lab.py architectures
```

[源码](../examples/lifelong_algorithms_lab.py)

## [实验设计：从更新正确到持续学习证据](../textbook/experiments.md)

标准库非平稳 bandit 的开发选择、独立测试、配对区间及失败诊断；不包含大型神经网络基准。

```bash
python examples/experiment_design_lab.py demo --tiny
```

[源码](../examples/experiment_design_lab.py)
