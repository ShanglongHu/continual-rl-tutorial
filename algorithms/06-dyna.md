[学习首页](../README.md) · [算法与代码目录](README.md)

# 06 从经验多学几次：模型、Dyna 与规划

先修：Q-learning；状态转移

问题：一条昂贵的真实经验，能否通过模型帮助更多价值更新？

真实 transition → 学习转移与奖励模型 → 模拟 transition → 规划更新

Model-free 不显式用环境模型做规划，不意味着没有任何预测。Dyna 的特别之处是同时学价值与模型：真实经验更新 Q，也更新模型；之后从模型生成虚拟经验，再交给同一个价值学习器。

符号：M(s,a)：学得的后继与奖励模型；k：每个真实步之后的规划更新数；θ：价值参数。下面的教学模型只适用于确定性环境。

## 把三个学习对象分开

行为策略决定采什么数据，价值函数决定当前偏好，模型预测动作后果。一个新奖励可以经真实 backup 改变局部 Q，也可以经模型上的多个 backup 传播到未实际重访的前序状态。

$$
(\hat s',\hat r,\hat d)\leftarrow M(s,a),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[\hat r+\gamma(1-\hat d)\max_bQ(\hat s',b)-Q(s,a)]
$$

## 模型更新与规划计算是两种成本

每个真实步之后增加 k 次模拟更新，可能提高样本效率，却用掉更多计算。报告中同时画真实环境步数与总更新量；只按 episode 比较容易把额外算力当成算法优势。

## 从 Dyna 接到更广的 model-based RL

Prioritized sweeping（优先更新） 选择最值得传播的 backup；MPC 每次用模型规划短期动作序列；学习潜在动力学的世界模型可用于想象训练。这些方法都涉及模型，但模型表示、如何使用、误差传播方式不同。不能把它们理解为一条自动升级的版本链。

## 手算例子

一个通道原来通向奖励 1，后来奖励变为 0。真实动作尚未再到终点时，模型仍可能保存奖励 1；做更多规划只会更自信地重复旧预测。问题可能是缺少新数据，而不是规划次数不够。

## 对照代码

```python
# control(..., experiment='dyna') 中的核心流程
q_step(q, s, a, r, ns, done, alpha, gamma)      # 真实更新
model[s, a] = (ns, r, done)                    # 学到的最近后果
for _ in range(planning):
    ms, ma = sample_seen_state_action(model)
    ns, r, done = model[ms, ma]
    q_step(q, ms, ma, r, ns, done, alpha, gamma)  # 模型更新
```

标准库包里的 control() 同时提供无规划 Q-learning 与 Dyna-Q，规划随机数单独管理；模型不会提前获知奖励切换。样例片段中 sample_seen_state_action 对应实际代码的 plan_rng.choice。

```sh
python3 examples/rl_foundations.py dyna --seeds 5 --episodes 800 --switch 400 --planning 10 --out dyna-run
```

把 --planning 改为 0，dyna_q 应逐点等于 q_learning。再增加规划数，观察变化前与变化后的差异，并核对 update_count 的增长。

## 怎样接到 CRL

CRL 的模型必须跟踪世界，也可能需要跟踪自身不断改变的技能。研究问题可以先缩小到“旧模型怎样拖慢适应”，再扩展到抽象技能模型。

适用条件：脚本存最近一次确定性转移，不估计概率模型；允许 episode reset，不是 single-life。Dyna-Q+ 的探索奖励未在此实现。

## 思考题

把规划次数翻倍后样本效率提高，能否说计算效率也提高？

提示：分别统计真实交互、模型调用与价值更新。

参考答案：不能直接说。样本效率、计算量和真实时间是不同指标；在机器人动作 deadline 下还需测每步延迟。

## 原始材料与代码

- [Sutton & Barto · 第 8 章](http://incompleteideas.net/book/the-book-2nd.html)：先看 Dyna，再比较环境变化与 Dyna-Q+。
- [Spinning Up · model-based 分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：区分 MPC、模型生成训练数据与规划嵌入策略。


---

[← 上一章](05-soft-control.md) · [下一章 →](07-skills.md)
