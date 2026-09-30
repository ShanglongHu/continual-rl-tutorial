# 实验与实现目录

[学习首页](../README.md) · [一键复现说明](reproducibility.md)

每次只选一个问题。以下命令从仓库根目录运行，输出文件名必须尚不存在。

## A. 五组基础算法实验

| 命令模式 | 看什么 | 关键边界 |
|---|---|---|
| prediction | MC、TD(0)、TD(λ) 相对解析真值的 RMSE | 固定策略、相同轨迹；不是控制回报 |
| control | SARSA 与 Q-learning 在奖励反转后的行为 | 表格、自然终止、允许位置 reset |
| policy | REINFORCE 与一步 actor–critic 的回报和动作概率 | γ=1 的表格 softmax，不是 PPO |
| dyna | 真实交互与模型 backup 的作用 | 确定性最近转移模型，不是深度世界模型 |
| consolidation | fine-tuning、rehearsal、二次 EWC 的权衡 | 确定性标量目标，不是深度 EWC 或 RL benchmark |

```sh
python3 examples/rl_foundations.py prediction --seeds 5 --episodes 1000 --out prediction-demo
python3 examples/rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --out control-demo
python3 examples/rl_foundations.py policy --seeds 5 --episodes 800 --alpha 0.05 --switch 400 --out policy-demo
python3 examples/rl_foundations.py dyna --seeds 5 --episodes 800 --switch 400 --planning 10 --out dyna-demo
python3 examples/rl_foundations.py consolidation --seeds 1 --strength 1 --out consolidation-demo
```

`--out` 是文件前缀，例如 `control-demo.csv` 与 `control-demo.html`。
`--switch 400` 表示完成第 400 个 episode 后反转，不是第 400 个环境步。
算法只看 transition，参数、模型不因切换清零。默认回报窗口每 20 个 episode 记录一次。

先做三个实现检查：

1. prediction 的 `--lam 0` 应让 td_lambda 与 td0 逐点相同。
2. dyna 的 `--planning 0` 应让 dyna_q 与 q_learning 逐点相同。
3. 开启 n 次规划时，Dyna 每个真实 transition 做 1+n 次价值更新；报告计算成本。

## B. 四组 CRL 诊断

```sh
python3 examples/crl_labs.py bandit --seeds 20 --steps 4000 --out bandit-demo.csv
python3 examples/analyze_crl.py bandit-demo.csv --out bandit-demo.html
python3 examples/crl_labs.py memory --seeds 1 --steps 1000 --out memory-demo.csv
python3 examples/crl_labs.py credit --seeds 5 --steps 4000 --out credit-demo.csv
python3 examples/crl_labs.py retention --seeds 1 --steps 1000 --out retention-demo.csv
```

| 模式 | 目标与检查 | 不要推断什么 |
|---|---|---|
| bandit | 跟踪反转奖励；同时看奖励、动作比例与 Q 值 | 不证明深度网络的可塑性 |
| memory | 平衡线索下，无记忆固定策略为 0.5，手工记忆 oracle 为 1 | 不是训练 RNN 的结果 |
| credit | 对照线性 SGD / IDBD 的可预测信号误差与步长 | 不是 NetworkIDBD；默认参数差异不等于公平排名 |
| retention | 旧目标退步时，新目标仍可快速学会 | 遗忘与可塑性损失不是同义词 |

memory 的当前统计由平衡协议决定，retention 是确定性反例；重复相同结果不增加独立证据。
只有 bandit CSV 使用 `analyze_crl.py`；其他三个诊断保留原始 CSV。

## C. 外部深度研究代码

| 想读哪类实现 | 先读本仓库 | 进入外部代码时先找什么 |
|---|---|---|
| DQN / Double DQN | [第 3 章](../algorithms/03-deep-value.md) | buffer、target、终止处理、梯度 |
| PPO / GAE | [第 4 章](../algorithms/04-policy.md) | rollout、advantage、minibatch、clip |
| TD3 / SAC | [第 5 章](../algorithms/05-soft-control.md) | 双 Q、actor 梯度、熵与动作变换 |
| 保留机制 | [第 8 章](../algorithms/08-retention.md) | task 边界、旧数据权限、参数重要性 |
| CBP / ReDo | [第 9 章](../algorithms/09-plasticity.md) | 特征统计、替换、优化器状态 |
| Stream-X / RTU | [第 10 章](../algorithms/10-streaming.md)、[第 11 章](../algorithms/11-state.md) | 单步顺序、状态与信用、归一化 |

官方入口与带问题的阅读说明在各章末尾。外部仓库不由本仓库 CI 安装或验证。
