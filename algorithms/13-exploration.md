[学习首页](../README.md) · [算法与代码目录](README.md)

# 13 世界会变化，为什么还要探索：计数奖励、RND 与学习进展

先修：ε-greedy、价值学习、预测误差

问题：已经很会做当前任务的 agent，怎样发现以前没有价值的新机会？

ε-greedy：保留尝试 → 访问计数：奖励少见 → RND：特征预测新奇 → 学习进展 / 技能探索

探索决定未来会获得什么经验，因此影响所有后续学习。持续世界里，曾经熟悉的状态也可能产生新奖励；只奖励“从没见过”还不够。这里把基础随机探索、内在奖励和自动课程联系起来，同时区分它们实际测量的东西。

符号：Nₜ(s)：到 t 为止的访问次数；bₜ：内在奖励；β：探索奖励权重；f：固定随机目标网络；f̂θ：可学习预测网络；ℓ：预测损失。

## ε-greedy 很简单，却提供不可缺少的对照

每一步保留 ε 的随机动作概率，可以持续尝试非贪心动作。但一步随机不保证探索到长距离目标；若去新区域需要十步协调行动，局部噪声可能效率很低。SAC 的熵奖励也不能自动代替时间上连贯的探索。

## 计数奖励：少见的状态值得再看

一个常见教学形式是访问越少、额外奖励越大。深度观测中精确计数不实用，可以使用密度模型或表示近似新奇度；伪计数并非简单统计图像哈希次数。

$$
\widetilde R_{t+1}=R_{t+1}^{\mathrm{ext}}+\beta b_t(S_{t+1}),\qquad b_t(s)=\frac{1}{\sqrt{N_t(s)+1}}
$$

这是解释计数奖励的简化形式，不是所有算法共享的精确公式。必须同时报告外在回报，避免只优化了自己加入的奖励。

## RND：预测一个固定随机网络的输出

目标网络 f 初始化后固定，预测器 f̂θ 在访问过的观测上学习拟合它。预测误差作为新奇信号；它不直接预测随机的下一帧，所以要区别于动力学预测误差。不过随机观测、泛化和预测器遗忘仍可能使新奇信号失真。

$$
b_t(o)=\left\|\widehat f_\theta(o)-f(o)\right\|_2^2,\qquad\theta\leftarrow\theta-\alpha\nabla_\theta b_t(o)
$$

原始 RND 还涉及归一化、外在/内在价值估计及训练细节；一行平方误差不是完整算法。

## 持续探索：从“误差大”到“还有进步空间”

学习进展比较一段时间内表现或预测损失的变化。比较时应控制评测分布；否则误差下降也可能只是采到了容易的状态。自动课程选择任务，options 组织连贯行为，它们和内在奖励是可以组合但需要分别检验的组件。

## 手算例子

状态已经访问 99 次，计数奖励为 0.1；一个新状态的奖励为 1。现在旧状态的外在奖励突然提高，历史计数仍不会自动增加它的新奇度。若探索完全停止，agent 甚至没有数据发现这次改变。

## 对照代码

```python
# RND 机制伪代码；target 参数固定，只更新 predictor
with no_grad():
    target_features = target_network(obs)
error = ((predictor(obs) - target_features) ** 2).sum(dim=-1)
intrinsic_reward = error.detach()
loss = error.mean()
predictor_optimizer.zero_grad()
loss.backward()
predictor_optimizer.step()
# 交给控制算法时，外在回报与内在奖励分别记录
```

先用 control() 比较持续探索概率，理解没看到新奖励时再好的更新也无从下手。然后读 OpenAI RND 的历史参考实现：它已归档，需按 README 另建兼容环境，不与教学包混装。

```sh
python3 examples/rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --epsilon 0.3 --out exploration-run
```

与 --epsilon 0.1 的 control-run 比较变化前回报、变化后恢复及真实交互量。再去 RND 检查目标网络是否真的冻结、predictor 的训练输入、奖励归一化与各自的 value head。

## 怎样接到 CRL

连接 Machado 的技能发现、自动课程和开放式学习。可从“旧区域价值改变时，新奇驱动还能否发现它”提出一个小实验，而不是只比较首次探索率。

适用条件：标准库实验只实现 ε-greedy，不实现 RND 或计数奖励。新奇度与学习进展都是代理信号，不能直接等同于长期控制价值或知识增长。

## 思考题

RND predictor 因持续训练遗忘了旧区域，会发生什么？

提示：内在奖励来自当前预测误差。

参考答案：重访旧区域时误差可能重新升高，把遗忘误认为新奇。需要区分真实新信息、表示漂移和预测器遗忘；这正是探索与可塑性、知识保留相互作用的接口。

## 原始材料与代码

- [Count-Based Exploration 原论文](https://arxiv.org/abs/1606.01868)：看密度模型如何构造伪计数，不只看奖励形状。
- [RND 原论文](https://arxiv.org/abs/1810.12894)：区分固定随机目标与学习中的预测器。
- [OpenAI RND 历史代码](https://github.com/openai/random-network-distillation)：已归档；用于对照原算法的奖励与训练循环。
- [课程学习综述](https://www.jmlr.org/papers/v21/20-212.html)：比较新奇、学习进展与任务选择。

---

[← 上一章](12-meta.md) · [进阶阅读 →](../docs/advanced-reading.md)
