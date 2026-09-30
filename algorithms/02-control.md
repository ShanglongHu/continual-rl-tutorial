[学习首页](../README.md) · [算法与代码目录](README.md)

# 02 从预测到决策：SARSA 与 Q-learning

先修：上一章 TD；ε-greedy

问题：要估计自己接下来实际怎么走，还是估计下次都选最优动作会怎样？

TD 预测 → 动作价值 Q → SARSA / Q-learning → 探索与控制

只有 V(s) 时，你知道一个状态大概好不好，却不能直接比较不同动作。用 Q(s,a) 预测先做 a 的长期后果，就能由价值改善行为。SARSA 与 Q-learning 的关键分歧不是用不用神经网络，而是 target 里的下一动作。

符号：Q(s,a)：动作价值；A′：行为策略真正选出的下一动作；ε：随机探索概率。行为策略生成数据，目标策略决定你想学习谁的价值。

## SARSA：把未来探索也算进来

按当前 ε-greedy 策略选出 A′，再用它的估值更新。算法名称对应 (S,A,R,S′,A′)。若探索可能走入危险区，SARSA 的估值会反映这种未来行为风险。

$$
\delta_t^{\mathrm{SARSA}}=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$

## Q-learning：target 假设下一步贪心

真实行动仍可探索，但 target 用下一状态最大 Q。它学习的目标与实际生成动作的探索策略不同，因此是 off-policy。这里的 off-policy 不等于必须使用 replay。

$$
\delta_t^{Q}=R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t),\qquad Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t^Q
$$

## Expected SARSA 是第三种 target

如果知道下一动作的概率，可以对 Q 加权平均，代替采样一个 A′。这减少了下一动作采样这一项噪声，但不消除环境随机性。

$$
y_t=R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)
$$

## 手算例子

R=0，γ=0.9，下一状态两个 Q 为 2、5。如果实际探索选了第一个动作，SARSA target 是 1.8，Q-learning 是 4.5。ε=0.1 的双动作 ε-greedy 下，Expected SARSA 为 0.9×(0.05×2+0.95×5)=4.365。

## 对照代码

```python
# q_step() 共用同一个更新；只改变 bootstrap target
boot = q[next_state][next_action]  # SARSA
# boot = max(q[next_state])       # Q-learning
target = reward if terminal else reward + gamma * boot
q[state][action] += alpha * (target - q[state][action])
```

读 q_step() 和 control()：SARSA 在更新前选择下一动作；Q-learning 用 max 构造 target。corridor() 决定奖励，切换标记不传入 q_step()。

```sh
python3 examples/rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --out control-run
```

生成回报、真实步数和更新次数三张图。第 400 个 episode 后左右终点收益交换；参数不清零。自然终止后位置重置，所以这是允许 episodic reset 的变化任务。

## 怎样接到 CRL

这组算法已经可以用于变化环境的基线。常数步长帮助更新旧估计，ε 帮助发现新收益；但是“知道变化时刻后手动清空 Q”是额外权限，不能偷偷加入。

适用条件：基础脚本不是 cliff-walking，也不保证展示 SARSA 与 Q-learning 的固定优劣排序。相同 episode 数下，轨迹长度可能不同，必须看真实交互量。

## 思考题

每步只用最新 transition 的 Q-learning，是 on-policy 还是 off-policy？

提示：看 target，而不是 buffer。

参考答案：仍然可以是 off-policy：行为含探索，target 却对下一动作取 max。数据是否回放与目标/行为策略是否一致，是两条不同的轴。

## 原始材料与代码

- [Sutton & Barto · 第 6 章](http://incompleteideas.net/book/the-book-2nd.html)：重点对照 SARSA、Q-learning 与 Expected SARSA 的备份目标。


---

[← 上一章](01-value.md) · [下一章 →](03-deep-value.md)
