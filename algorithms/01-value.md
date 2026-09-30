[学习首页](../README.md) · [算法与代码目录](README.md)

# 01 价值从哪里来：Bellman、MC、TD 与资格迹

先修：状态、奖励、期望与加权平均

问题：还没走到终点，能否用刚发生的一步改善预测？

Bellman 递推 → MC：完整结果 → TD：一步自举 → 多步 / TD(λ)

先固定策略，只问“照这样走，未来能得到多少奖励”。动态规划用已知模型对所有可能后果取期望；Monte Carlo（MC）用实际完整轨迹；TD 用一步实际奖励接上对未来的估计。它们首先是不同的价值更新方式，不是三种不同的奖励目标。

符号：Sₜ、Aₜ、Rₜ₊₁：当前状态、动作、下一次奖励；γ：折扣；V：状态价值估计；α：学习步长。真实终止状态的后继价值为 0。

## 先定义要预测的量，再写 Bellman 关系

回报是未来奖励的折扣和。拆出第一项，就把长时间预测变成“即时奖励 + 下一状态价值”。Bellman 等式描述正确答案应满足的关系；它本身还没有告诉你怎样从有限数据学习。

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad v^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma v^\pi(S_{t+1})\mid S_t=s]
$$

## 有模型时：策略迭代与价值迭代

若已知转移及奖励模型，策略迭代交替做“评估当前策略”和“对价值贪心改进”。价值迭代将二者合成最优 Bellman backup。下面对下一状态求精确期望；MC/TD 则从采样经验获得更新信号。实际复杂环境通常没有这样的完整模型。

$$
V_{k+1}(s)=\max_a\sum_{s'}p(s'|s,a)\left[r(s,a,s')+\gamma V_k(s')\right]
$$

这里 r 是给定转移的期望奖励。标准保证需要有限 MDP、折扣 γ<1 等条件；无折扣任务要另检查终止性。

## MC 与 TD 的区别在 target

二者都用“估计 ← 估计 + 步长 × 误差”。MC 等完整轨迹结束，用 Gₜ；TD(0) 每步即可用 R+γV(S′)。TD 的 target 包含自己的估计，称为自举：更及时，但也会传递估计误差。

$$
\begin{aligned}V(S_t)&\leftarrow V(S_t)+\alpha[G_t-V(S_t)]&&\text{MC}\\V(S_t)&\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]&&\text{TD(0)}\end{aligned}
$$

## 多步与资格迹：让奖励影响更早的状态

n-step target 用 n 个真实奖励再自举。资格迹则给过去参与过的状态或特征保留一份衰减责任记录。下面是表格、on-policy、accumulating trace；每个 episode 开始清零。

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n}),\quad e_t=\gamma\lambda e_{t-1}+\mathbf1_{S_t},\quad V\leftarrow V+\alpha\delta_t e_t
$$

接近终点时按真实剩余奖励截断。λ=0 回到 TD(0)；在线 accumulating traces 的 λ=1 不应不加条件地说成与逐次 MC 完全相同。

## 手算例子

一条两步轨迹奖励为 0、1，γ=0.9，V(s₀)=0.2，V(s₁)=0.4。s₀ 的 MC target 为 0.9；第一步 TD target 为 0.36。α=0.1 时分别得到 0.27 和 0.216。目标都关心未来奖励，只是使用的证据不同。

## 对照代码

```python
# 一条 transition；terminal 是真正终止，而非任意日志窗口结束
target = reward if terminal else reward + gamma * values[next_state]
delta = target - values[state]
trace = [gamma * lam * e for e in trace]
trace[state] += 1
for s in nonterminal_states:
    values[s] += alpha * delta * trace[s]
```

打开 examples/rl_foundations.py，先读 prediction()。它先生成固定策略的相同轨迹，再分别交给 every_visit_mc、td0 和 td_lambda，避免把策略不同误认为预测规则不同。

```sh
python3 examples/rl_foundations.py prediction --seeds 5 --episodes 1000 --alpha 0.1 --out value-run
```

打开 value-run.html：纵轴是相对解析真值 v(i)=i/6 的 RMSE，不是训练回报。将 --lam 改为 0，td_lambda 与 td0 的输出应一致。

## 怎样接到 CRL

常数步长、即时更新和迹，是持续预测的基础。但平稳随机游走上能预测，并不证明面对长期变化也能跟踪；下一步改变目标时，应保留同一学习器而非重训。

适用条件：这里是有限状态、固定策略、自然终止任务。用神经网络、off-policy 数据或递归状态时，不能直接沿用表格收敛直觉。

## 思考题

γ=0.9，下一状态确实终止且奖励为 1；它的数组里残留 V=100。target 应为多少？

提示：终止之后没有这个任务的后续回报。

参考答案：为 1，而不是 91。把终止标记与时间上限截断混淆，可能系统性改变学习目标。

## 原始材料与代码

- [Sutton & Barto · 第 4–7、12 章](http://incompleteideas.net/book/the-book-2nd.html)：先比较 backup 目标，再看 traces；不用先学神经网络。


---

[← 目录](README.md) · [下一章 →](02-control.md)
