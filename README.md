# Continual RL Tutorial · 强化学习：从基础到持续研究

[![Teaching checks](https://github.com/ying-wen/continual-rl-tutorial/actions/workflows/test.yml/badge.svg)](https://github.com/ying-wen/continual-rl-tutorial/actions/workflows/test.yml)

从表格预测与控制出发，经过函数逼近和深度强化学习，进入持续学习的目标、状态、知识、信用、技能与规划。

**19 章基础分册 · 22 章 CRL 教材 · 8 课导论 · 配套实现、数值检查与实验手册**

[在线阅读](https://yingwen.io/zh/continual-rl/) · [基础目录](foundations/README.md) · [CRL 教材](textbook/README.md) · [实验与代码](docs/experiments.md) · [研究问题](docs/research-atlas.md)

Tutorials are currently in Chinese. Original code is MIT; original tutorial text is CC BY 4.0. Third-party works keep their own licenses.

## 从哪里开始

| 已有基础 | 阅读顺序 | 完成时应能做什么 |
|---|---|---|
| 认识状态、动作、奖励 | [表格方法 / Part I](foundations/tabular/README.md) | 从回报写出 Bellman 关系，区分预测、控制和规划，实现更新并核对数值 |
| 学过 Sutton Part I | [函数逼近 / Part II](foundations/approximation/README.md) | 区分真梯度与半梯度，解释投影、离策略不稳定、资格迹和策略梯度 |
| 能运行 DQN 或 PPO | [现代 DRL](foundations/deep/README.md) | 追踪 target、梯度、数据与状态的完整时序，检查训练与算法公式是否对应 |
| 准备做 CRL 研究 | [22 章教材](textbook/README.md) → [研究地图](docs/research-atlas.md) → [实验手册](docs/experiment-handbook.md) | 为一个机制提出竞争解释，建立对照，保留完整运行并界定结论 |

Part II 不是可选的深度学习附录。状态共享参数之后，表格方法的保证不再自动成立。线性预测、特征构造、平均奖励、离策略方法、资格迹与策略梯度构成后续章节的共同基础。

每章按问题设定、必要符号、推导、执行顺序、数值例、实现、限制和练习组织。原论文与作者代码放在正文之后，供进一步核对。正文为原创解释，不是 Sutton 教材或 Spinning Up 的翻译复制。

## 先运行数值与实现检查

需要 Python 3.10+。标准库教学部分无需 GPU、pip 安装或数据集下载。

```bash
git clone https://github.com/ying-wen/continual-rl-tutorial.git
cd continual-rl-tutorial
python3 scripts/test_examples.py
python3 -m unittest discover -s tests -v
```

每个章节给出独立命令。所有命令按仓库根目录编写。建议先改一个断言或一条短轨迹，观察实现能否识别错误，再开始长训练。

要生成原有的 CSV 与 HTML 小实验报告：

```bash
python3 scripts/run_all.py --quick --out results/first-run
```

打开 `results/first-run/bandit.html`。其余报告解释预测、控制、策略更新、Dyna 和保留。运行清单记录版本、源码和输出摘要。已有输出目录不能覆盖；再运行时换新目录。

## 代码分为哪些层次

| 层次 | 入口 | 范围 |
|---|---|---|
| 表格方法 | [tabular_textbook_lab.py](examples/tabular_textbook_lab.py) | 小 MDP、bandit、DP、MC、TD、多步与规划；解析或有限轨迹核对 |
| 函数逼近 | [approximation_textbook_lab.py](examples/approximation_textbook_lab.py) | 线性特征、半梯度、投影、离策略、平均奖励、迹和策略梯度的受控例子 |
| 深度更新核 | [deep_textbook_lab.py](examples/deep_textbook_lab.py) | DQN、GAE、TRPO/PPO、TD3、SAC 的目标、数值与时序检查 |
| 神经训练 | [deep_textbook_train.py](examples/deep_textbook_train.py) | 可选 PyTorch CPU；小环境中的 DQN/PPO 训练及梯度检查 |
| CRL 机制 | [逐章实验](docs/experiments.md) | GVF、状态、平均奖励、信用、元学习、options、模型规划及长期学习机制 |
| 实验语义 | [algorithm_testing_lab.py](examples/algorithm_testing_lab.py) | 终止与截断、策略目标、option 时钟和完整检查点状态 |

可选神经实现：

```bash
python3 -m pip install -r examples/deep_requirements.txt
python3 examples/deep_textbook_train.py test
```

这些检查不等于 Atari、MuJoCo、单生命期持续世界或原论文性能复现。TRPO、TD3、SAC 等方法的更新核与完整训练工程分别说明。完整作者实现入口见 [实现导读](docs/implementations.md) 与各章文献。

## 从学习转向研究

[研究与前沿](docs/research-atlas.md) 按问题连接机制、假设、原文、作者代码和可检验实验。[实验与算法测试手册](docs/experiment-handbook.md) 讨论对照、预算、开发与确认、统计单位、失败、恢复和持续交互。

需要固定协议、执行小矩阵、保存全部运行并审计结果时，使用公开的 [RL Research Workbench](https://github.com/ying-wen/rl-research-workbench)。先跟做它的 [完整例子](https://github.com/ying-wen/rl-research-workbench/blob/main/docs/worked-example.md)。教程代码与 Workbench 不自动共享运行接口；新增方法需要经过适配与验证。

- [统一学习路线](docs/learning-route.md)
- [课程、论文与学者的章节对应](docs/chapter-companions.md)
- [实验协议](docs/protocol.md)
- [八课 CRL 导论](tutorials/README.md)
- [兼容原链接的算法目录](algorithms/README.md)

## 内容同步与贡献

网站与仓库共享公开导出清单。代码按原字节同步；Markdown 转换内部链接和运行路径。CI 核对摘要，防止网页与 GitHub 长期分叉。维护过程见 [同步说明](docs/site-sync.md)。

欢迎提交可以复核的公式反例、运行失败或解释改进。请写明章节、命令、版本、期望与实际结果。修改前阅读 [贡献指南](CONTRIBUTING.md)。方法效果、实现正确和机制解释分别验收，负结果也应保留。

[代码 MIT](LICENSE-CODE) · [原创教程 CC BY 4.0](LICENSE-DOCS.md) · [许可范围](LICENSE) · [来源](docs/sources.md)

作者：[Ying Wen / 温颖](https://yingwen.io/)。
