# Continual RL Tutorial · 持续强化学习：从基础到研究

[![Teaching checks](https://github.com/ying-wen/continual-rl-tutorial/actions/workflows/test.yml/badge.svg)](https://github.com/ying-wen/continual-rl-tutorial/actions/workflows/test.yml)

面向刚学过基础 RL，以及已熟悉深度 RL、准备进入持续强化学习研究的读者。
从一个可解释的问题开始，连起算法、公式、代码和实验，最后选择一个具体研究方向。

**8 课入门 · 13 章算法 · 9 组教学实验 · Python 标准库 · 自动生成图表与复现记录**

Tutorials are currently in Chinese. Teaching code uses only the Python standard library.
Original code: **MIT**. Original tutorial text: **CC BY 4.0**.

## 先选一条路线

| 你现在的起点 | 第一站 | 接着做什么 |
|---|---|---|
| 刚学过状态、动作、奖励和价值函数 | [第 1 课：为什么需要 CRL](tutorials/01-lifetime.md) → [第 2 课：跟踪变化](tutorials/02-tracking.md) | [做第一个完整实验](docs/first-experiment.md) |
| 学过 RL，但算法之间还没连起来 | [MC / TD](algorithms/01-value.md) → [SARSA / Q-learning](algorithms/02-control.md) → [策略梯度](algorithms/04-policy.md) | 按需补 DQN、PPO、SAC、Dyna |
| 已熟悉 DQN / PPO，想做研究 | [五份共同阅读材料](docs/quick-reading.md) → [分方向进阶阅读](docs/advanced-reading.md) | 选一个失败模式，写协议，跑最小对照 |

不要把所有章节都当先修。每章都有：问题、适用条件、公式解释、手算例子、代码对应、检查点、思考题及答案。

## 五分钟跑通代码

推荐 Python 3.10 或更新版本；无需 GPU、PyTorch、pip 安装或数据下载。
Windows 若使用 `python` 或 `py -3`，替换命令中的 `python3` 即可。

```sh
git clone https://github.com/ying-wen/continual-rl-tutorial.git
cd continual-rl-tutorial
python3 scripts/run_all.py --quick --out results/first-run
```

打开 `results/first-run/bandit.html` 看第一个 CRL 例子；
再看 `prediction.html`、`control.html`、`policy.html`、`dyna.html`、`consolidation.html`。
这些都是可直接打开的独立 HTML 文件。

`--quick` 只检查能否运行。完成正文实验时，使用新的目录运行完整教学配置：

```sh
python3 scripts/run_all.py --out results/teaching-run
python3 -m unittest discover -s tests -v
```

输出包括 9 个 CSV、6 个 HTML 和 `manifest.json`。
Manifest 记录命令、Python 版本、系统类型、Git commit、源码 SHA-256 和结果 SHA-256。
所有脚本都保护已有结果；再次运行请换一个 `--out` 名称，不要删掉前一次结果来“修正”曲线。

## 算法主线：CRL 改变了基础算法的什么？

```text
价值学习：Bellman / MC / TD → SARSA / Q-learning → DQN
策略学习：REINFORCE → actor–critic / GAE → PPO；连续控制接 TD3 / SAC
模型与知识：Dyna → 技能与技能模型；SR / SF → 迁移

放到持续的生命期中，再问：
├─ 旧知识怎样保留？                Replay / EWC / 蒸馏
├─ 新知识还能学进去吗？            ReDo / Continual Backprop
├─ 每一步来得及并且稳定地更新吗？   Traces / IDBD / Stream-X
├─ 历史怎样变成有用状态？          GVF / RNN / RTU
└─ 下一步学什么、复用什么？        Options / Meta-RL / 课程 / 探索
```

这是学习依赖与方法接口，不是一条所有方法相互替代的历史链。

- [八课入门目录](tutorials/README.md)：从概念、现象到实验协议。
- [十三章算法目录](algorithms/README.md)：从更新式到实现。
- [实验与实现目录](docs/experiments.md)：每个命令做什么、测什么、没有实现什么。
- [十条研究路线](docs/advanced-reading.md)：代表论文、作者、代码、benchmark、课程讲座与跨学科材料。
- [复现约定](docs/reproducibility.md)与[实验协议模板](docs/protocol.md)。

## 这里提供什么代码？

| 实验包 | 实际可运行的实现 | 输出 |
|---|---|---|
| [rl_foundations.py](examples/rl_foundations.py) | MC / TD(0) / TD(λ)、SARSA / Q-learning、REINFORCE / 一步 actor–critic、Dyna-Q、标量 replay/EWC 例子 | CSV + HTML |
| [crl_labs.py](examples/crl_labs.py) | 奖励跟踪、记忆信息条件、线性 IDBD 信用分配、遗忘反例 | CSV |
| [analyze_crl.py](examples/analyze_crl.py) | 将配套 bandit CSV 生成为四图报告 | HTML |

DQN、PPO、SAC、ReDo、完整 CBP、RTU、RND 等章节提供推导和外部实现的读码路线，
**本仓库没有声称实现或复现这些深度研究系统**。
表格控制实验允许自然终止与 episode reset，不是 single-life benchmark。
memory 是手工 oracle 诊断；consolidation 与 retention 是确定性教学反例。

## 从教学实验进入研究

先写一个可证伪的问题，例如“旧模型是否会延长变化后的错误价值传播”，
再固定信息权限、计算预算、调参范围和指标。
[第一份实验教程](docs/first-experiment.md)带你走完
“公式 → 运行 → 看图 → 单因素对照 → 一页结论”。

研究路线中的外部仓库有独立的依赖和许可；不要把它们全部安装到一个环境。
本仓库的 CI 检查教学机制、文档链接和可执行流程，不给任何方法作普遍优越性保证。

## 贡献、来源与许可

欢迎通过 Issues 报告公式、链接、运行或解释上的问题；修改前请读 [CONTRIBUTING.md](CONTRIBUTING.md)。
引用研究结论时请引用原论文；使用本教程或代码时保留来源，并记录具体 commit。

- [来源与材料边界](docs/sources.md)
- [代码 MIT](LICENSE-CODE)
- [原创教程 CC BY 4.0](LICENSE-DOCS.md)
- [许可适用范围](LICENSE)

作者：[Ying Wen / 温颖](https://yingwen.io/)。
