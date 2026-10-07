# Monte Carlo：完整经历、重复访问与离策略评价

不知道模型时，可以将完整回报作为样本。需要规定重复访问如何计数、数据由谁生成，以及目标策略能否被行为覆盖。

## 本章内容

- 实现 first-visit、every-visit 和 ε-soft MC 控制。
- 推导轨迹与逐决策重要性采样，区分 ordinary IS 与 weighted IS。
- 识别终止、覆盖、方差和策略变化的边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

模型未知，但能观察直到真实终止的完整回合。首先估计固定策略的期望回报。

### 给定条件与符号

- 给定目标策略 $\pi$，$v_\pi(s)=\mathbb E_\pi[\sum_{k\ge0}\gamma^kR_{t+k+1}\mid S_t=s]$。
- 误差权重 $d(s)\ge0$ 且 $\sum_sd(s)=1$；权重指定在哪些状态上评价估计。
- 回合终止时间 $T$，完整回报 $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$；数据由行为策略 $b$ 产生。

### 需要求解的对象

用采样回报估计 $v_\pi$ 或 $q_\pi$；控制时还需改变策略并重新保证动作覆盖。

### 信息与数据权限

不能对未知环境求精确期望；离策略评价要求可计算目标与行为的动作概率。

$$
\min_v\sum_s d(s)[v(s)-v_\pi(s)]^2,\qquad \rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}
$$

第一式是预测误差；第二式是变换轨迹分布的权重，不是另一种外部奖励。对同策略回合，权重为一。

### 成立条件与解的含义

- 回合几乎必然终止并满足所用估计器的可积条件。
- 目标可能发生的动作必须有行为支持；有限方差需更强条件。

判断准则：分别检查估计偏差、方差和有效样本量；不要只看权重归一化后曲线平滑。

### 适用边界

- 在没有覆盖的数据中识别未观察行为的价值。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [价值预测与资格迹](../../textbook/value.md)：本章要求收到完整终止回合，使用不自举的回报样本估计同一个固定策略价值；改变的是可用数据与等待时刻。

- 改变信息或数据协议 · [流式更新与稳定性](../../textbook/streaming.md)：本章允许等待完整回报，可能超出严格逐步内存或延迟约束。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

只能看到随机回报；行为策略不同还会改变样本分布。

### 本章的核心思路

先用完整回报避免 bootstrap 偏差，再用概率比校正行为分布。

1. [确定一次状态访问怎样计数](monte-carlo.md#lesson-derive)：First-visit 与 every-visit 使用不同相关样本，不能混用计数和更新。

2. [由轨迹分布推出校正](monte-carlo.md#mc-importance)：相同环境核抵消后留下动作概率比，覆盖条件由分母直接给出。

3. [在无偏与稳定性之间选择](monte-carlo.md#mc-estimators)：普通重要性采样与自归一化估计器有不同有限样本性质。

结论与条件：一致性需要覆盖和相应大数条件；普通 IS 可能方差极大，weighted IS 一般有有限样本偏差。

### 相关方法改变了什么

- MC / TD：MC 等真实未来；TD 用当前预测补足未观察未来。

- Ordinary / weighted IS：分母分别用样本数和权重总和，不是可互换的归一化细节。


<a id="lesson-setting"></a>

## 1 · 用完整经历预测

已知模型使我们能对所有后果求平均；真实学习器往往只有自己经历的一条路径。一个直接办法是等这条经历结束，再用实际获得的总回报回答“当时处境有多好”。Monte Carlo 因而不必先学完整转移模型，也不必相信后继状态的价值估计。它把问题转成回报的统计估计，同时留下等待完整结果的代价。

给定固定策略 $\pi$，从规定初始分布收集到真实终止 $T$ 的经历。模型未知，只保存状态、动作和奖励。要求终止与回报可积条件足以使目标期望存在；无限不终止流不能等待一个永远不会出现的完整回报。

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

MC 使用实际完整结果，不接后继价值估计，因此没有 bootstrap；代价是等待和整个未来的随机性。

先固定策略研究预测，再加入行为改善。控制会改变生成下一条经历的策略，也改变需要评价的价值。实现里应能指出某个回报由哪一版策略生成，不能将数据与目标的区别隐藏在训练循环中。

<a id="lesson-derive"></a>

## 2 · First-visit 与 every-visit

同一状态可能在一条经历中出现多次。first-visit 每个状态每条经历只取时间上第一次访问后的回报；every-visit 纳入每次访问后的回报。从终点反向计算回报时，第一次遇到状态是最后访问，不能直接当作 first-visit。

$$
N(s)\leftarrow N(s)+1,\qquad V(s)\leftarrow V(s)+\frac{G-V(s)}{N(s)}
$$

N 只统计实际纳入的回报。first-visit 每条经历最多增加一次，every-visit 可以增加多次。

经历内多个回报相关，不宜把它们当独立样本来构造误差条。固定策略、独立 episode 与适当访问条件支持相应一致性结果；有限样本中两种估计可明显不同，样本数量也不等于有效独立信息量。

<a id="mc-control"></a>

## 3 · 动作价值与探索覆盖

$$
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a],\qquad \pi(a\mid s)=\frac{\epsilon}{|\mathcal A(s)|}+(1-\epsilon)\mathbf1\{a=g(s)\}
$$

按状态动作对汇总回报，g 是当前 Q 的贪心动作。此式定义 ε-greedy 策略，它属于 ε-soft 策略类。

Soft 只要求各合法动作概率为正；$\epsilon$-soft 进一步要求 $\pi(a\mid s)\ge\epsilon/|\mathcal A(s)|$，其中固定 $0<\epsilon\le1$。在同一个概率下界约束内，$\epsilon$-greedy 把剩余概率全部分给一个最大动作。这个约束决定了下面能够证明哪一种改善。

$$
\begin{aligned}\sum_a\pi'(a\mid s)q_\pi(s,a)&=\frac\epsilon K\sum_aq_\pi(s,a)+(1-\epsilon)\max_aq_\pi(s,a)\\&\ge\sum_a\pi(a\mid s)q_\pi(s,a)=v_\pi(s),\qquad K=|\mathcal A(s)|.\end{aligned}
$$

旧策略 π 满足同一个 ε-soft 下界，新策略 π′ 对准确的 qπ 作 ε-greedy 选择。减去各动作的 ε/K 后，旧策略仅剩总量 1−ε 的非负概率，分给最大动作至少同样好。再用策略改善定理，可得逐状态 vπ′≥vπ。

这就把探索约束接回了策略改善。有限折扣情形下，准确评价与此类改善若达到不动点，得到的是 ε-soft 策略类内的最优策略；无折扣回合式情形还需保证有关策略适当终止。每条经历后只做一次有噪声的 Q 更新，并不保证当次行为收益单调提高。固定 ε 的价值包含探索后果，不能直接写成无约束 q*；让 ε 衰减时还要另查无限访问等条件。

Exploring starts 假定可从任意相关状态动作对以正概率开始经历，这是一种很强的重置权限。ε-soft 减少了对该权限的依赖，但动作正概率不保证有限预算里能到达每个深层状态。

<a id="mc-importance"></a>

## 4 · 从轨迹概率比推导离策略评价

数据由行为策略 b 产生，却要评价目标策略 $\pi$。轨迹概率由动作概率和环境转移概率交替相乘。在相同环境中取两种策略的轨迹概率比，环境项逐项相消，只剩动作概率。

$$
\rho_{t:T-1}=\prod_{k=t}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)},\qquad \mathbb E_b[\rho_{t:T-1}G_t\mid S_t=s]=v_\pi(s)
$$

状态价值的比率包含当前动作；估计已经条件于当前状态动作的 qπ 时，从 t+1 开始校正。

$$
\pi(a\mid s)>0\Longrightarrow b(a\mid s)>0
$$

覆盖使目标的可能后果能在行为数据中出现。重要性采样不能制造不存在的数据。

必须保存采样时的行为概率；更新策略后重新计算旧动作概率会改变权重。行为若依赖历史，应使用对应历史下的条件概率。长轨迹的比率乘积可能造成巨大方差和数值下溢，提高浮点精度不能消除统计方差。

<a id="mc-estimators"></a>

## 5 · Ordinary 与 weighted IS

$$
\hat v_{\rm OIS}=\frac1n\sum_{i=1}^{n}\rho_iG_i,\qquad \hat v_{\rm WIS}=\frac{\sum_i\rho_iG_i}{\sum_i\rho_i}
$$

普通 IS 除以样本数，加权 IS 除以权重和。所有权重为零时后者无定义，代码返回 None。

这里普通 IS 的有限样本无偏性针对固定数目的合适回报样本，例如固定策略下独立收集的 first-visit 回报，并要求覆盖与可积性。若固定 episode 数，再把全部 every-visit 回报除以随机访问总数，估计一般有偏；随机分母与回报相关，不能直接套用固定样本数的均值证明。两种访问规则在适当条件下都可一致。

一个不需要离策略的反例就能说明区别。只有一个非终止状态，每步奖励为1，并以 $1/2$ 概率终止；令 $\gamma=1$。回合长度 $T$ 满足 $\mathbb E[T]=2$，真实价值也为2。只收集一个回合时，first-visit 估计为 $T$，every-visit 估计为 $(T+1)/2$，两种估计的期望分别为2和1.5。所有重要性比都为1，偏差仍然存在。

加权 IS 是随机分子除以随机分母，有限样本一般有偏；权重和为正时，它是已观察回报的凸组合，因而不会超出这些回报的范围。普通 IS 没有这个范围保证，二阶矩可能很大甚至不存在。加权 IS 在适当条件下可一致，并常有较小方差，但一次估计接近答案不足以建立普遍优势。

$$
C\leftarrow C+\rho,\qquad V\leftarrow V+\frac{\rho}{C}(G-V)
$$

将新的分子与分母展开即可得到增量式。C 是累积权重而非访问数。权重为零不改变结果；分母尚为零须单独处理。

还可以利用奖励发生的先后顺序缩短比率乘积。一个奖励已经出现后，后续动作不会改变它；在给定过去的条件下，每个后续动作比率的期望为1。对未来逐次取条件期望，就能删除这个奖励之后的比率。这一步用的是条件期望，不要求各时刻动作或奖励独立。

$$
\widetilde G_t=\sum_{k=t}^{T-1}\gamma^{k-t}\rho_{t:k}R_{k+1},\qquad \mathbb E_b[\widetilde G_t\mid S_t=s]=v_\pi(s).
$$

逐决策重要性采样让第 k+1 个奖励只乘从 t 到 k 的动作比率。先在有界有限时域下推导；随机终点还需支持及换序所需的可积条件。对动作价值，当前动作已被条件化，比率从 t+1 开始，空乘积为1。

例如奖励只在第一步出现，第二步仅作一个随机动作再终止。普通轨迹 IS 仍把第二动作的比率乘到第一奖励上；逐决策 IS 则删除这项无关随机性。它保持目标期望，并可能降低方差；有多个相关奖励时不能由此推出整体方差总是更小，也不能随意再除以一个轨迹权重和而仍宣称无偏。这里增加的是估计原理，现有教学脚本仍运行整轨迹 IS。

<a id="experiment-weighted_is"></a>

### 实验：实验 · 自归一化降低波动时，也改变了有限样本性质

同样的完整回合、同样的精确概率比，只改分母，为什么会产生不同的估计误差？

**环境与可用信息。** 每回合恰好三步。行为策略两个动作各选 0.5，目标策略选 0.2 和 0.8。两个动作的 Bernoulli 奖励均值为 0.1 和 0.9，γ=0.9。目标策略每步期望奖励为 0.74，起点真值为 2.0054。

**设置。** 种子 0–4，各采集 1200 个完整回合，即 3600 个真实转移。两种估计器使用相同回合。整轨迹权重是三个精确动作概率比的乘积；ordinary IS 用加权回报之和除以回合数，weighted IS 则除以权重之和。累积量从零开始。

**检验的机制。** Ordinary IS 在支持条件成立时对目标回报无偏，但随机总权重会带来波动。Weighted IS 将权重归一为一，代价是有限样本通常有偏。只有一个正权重样本时，它甚至完全约掉权重，直接返回那条行为轨迹的回报。

**测量。** 纵轴是每次运行的回报估计与 2.0054 的绝对差，再对五种子取平均。它不是带符号统计偏差，也不是均方误差。原始日志中的 estimate 可以用于查看误差方向。

```bash
python3 implementations/extended_classic/weighted_is.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/weighted_is/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 episodes。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 第 1200 回合，weighted IS 的平均绝对误差为 0.0190，ordinary IS 为 0.0554。自归一化在这组短回合上更接近真值，但不能由五条绝对误差曲线证明它无偏或在所有长度上更好。

**结论边界。** 行为概率已知，且每个目标动作都有正行为概率。三步回合不能展示长轨迹比率乘积的全部困难；本实验也不包含控制策略学习。

**继续实验。** 先计算单个回合时两种估计器的期望，区分偏差与方差。然后增加回合长度，记录权重总和、最大权重和有效样本量，并始终与解析目标真值比较。

[源码](../../implementations/extended_classic/weighted_is.py) · [逐种子记录](https://yingwen.io/crl-code/results/weighted_is/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/weighted_is/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/weighted_is/curves.json)

<a id="mc-algorithm"></a>

## 6 · 完整算法步骤

**算法：算法伪代码**

1. 规定目标 π、行为 b、回报与真实终止。
1. 收集一条完整经历，保存状态、动作、奖励、采样时的行为概率。
1. 从终点向前递推每个时刻的 G。
1. 按时间正向选择 first-visit，或保留 every-visit。
  1. On-policy：更新相应状态或状态动作的回报均值。
  1. Off-policy：构造对应起点的比率，再更新普通或加权估计。
1. 若做控制，用新 Q 改善行为，下一条经历采用新策略。

回报反向递推与首次访问正向选择分开，避免 last-visit 错误。固定目标的 IS 评价也不应在同一批权重计算中偷偷改变目标策略；变化目标需要另外定义每个估计的含义。

离策略控制还需要把评价与改善结合。若目标是确定性贪心策略，一旦经历中出现不符合目标的未来动作，该动作之前的整段普通重要性比率就变成零。这解释了反向加权控制常在行为偏离目标处停止传播，也解释了长经历中有效回报可能很少。这里只实现固定目标 IS 评价与 on-policy MC 控制，二者没有被混作一个离策略控制程序。

<a id="lesson-example"></a>

## 7 · 手算、实际采样与源码

两步经历都访问 S，奖励为 1、2，γ=1。两次回报是 3、2；first-visit 得 3，every-visit 得 2.5。代码专门检查这一差异。

一个决策的离策略实验：动作成功率 (.2,.8)，行为均匀，目标概率 (.2,.8)，真实价值为 .68。10000 个样本得到 ordinary≈.68004、weighted≈.685303。两者估计同一个目标；误差来自采样与归一化。

控制例为 A 退出得 .2，或零奖励到 B；B 两动作得 1、−1 后终止。ε=.1 时 B 好动作概率 .95，所以 A 继续的探索策略价值为 .81，默认 MC 约 .806214；纯贪心价值为 .9。

<a id="lesson-code"></a>

## 8 · 实验观察与估计诊断

访问选择、概率连乘、IS 估计与实际 ε-soft 控制循环。公共采样和模型函数包含在完整脚本中。

```python
def mc_visits(states, rewards, gamma=1.0, first_visit=True):
    if len(states) != len(rewards):
        raise ValueError("states exclude terminal")
    totals, counts, seen = {}, {}, set()
    for state, target in zip(states, returns(rewards, gamma)):
        if first_visit and state in seen:
            continue
        seen.add(state)
        counts[state] = counts.get(state, 0)+1
        totals[state] = totals.get(state, 0.)+target
    return {s: totals[s]/counts[s] for s in totals}

def importance_ratio(target_probabilities, behavior_probabilities):
    if len(target_probabilities) != len(behavior_probabilities):
        raise ValueError("probability sequences must align")
    weight = 1.
    for pi, b in zip(target_probabilities, behavior_probabilities):
        if b <= 0:
            raise ValueError("sampled action needs positive behavior probability")
        weight *= pi/b
    return weight

def is_estimates(samples):
    numerator = sum(g*w for g, w in samples)
    weight = sum(w for _, w in samples)
    return numerator/len(samples), (numerator/weight if weight else None)

def mc_control(episodes=6000, seed=7, epsilon=0.1):
    rng = random.Random(seed)
    q = {s: [0.]*len(CONTROL_MODEL[s]) for s in CONTROL_MODEL}
    counts = {s: [0]*len(q[s]) for s in q}
    for _ in range(episodes):
        trajectory, rewards, state = [], [], 0
        while state is not None:
            action = choose(epsilon_probs(q[state], epsilon), rng)
            reward, sp = step(CONTROL_MODEL, state, action, rng)
            trajectory.append((state, action))
            rewards.append(reward)
            state = sp
        seen = set()
        for (s, a), target in zip(trajectory, returns(rewards, .9)):
            if (s, a) not in seen:
                seen.add((s, a))
                counts[s][a] += 1
                q[s][a] += (target-q[s][a])/counts[s][a]
    return q
```

先检查重复访问算例，再运行随机离策略评价。将行为对目标偏好动作的概率逐渐减小，并重新计算采样时的概率比，观察非零高权重样本变少。记录最大权重和权重和，不只记录最后估计。若直接把行为概率改为零，这不是方差变大的极限实验，而是破坏了支持条件，无法再从这份数据评价原目标。

控制实验中把 ε 从 .1 改为 .2，B 的坏动作概率变为 .1，因此 A 继续的行为价值由 .81 变为 .72。这个解析变化可以检验程序是否确实在评价含探索的行为。每次修改均应重置统计表；把不同设定的数据累加进同一均值，会使误差混入目标变化。

<a id="lesson-branches"></a>

## 9 · 边界与持续学习

完整 MC 不适合直接等待无限生命的最后结果。可以改成明确的有限时域目标，或在窗口末尾 bootstrap；后一种已经进入多步 TD。持续变化时，旧回报可能来自不同后果分布，等权保留全部历史未必适合跟踪。

回放可能产生 off-policy 数据，但不意味着所有回放方法都乘完整轨迹比率。一步 Q-learning、树备份和其他修正具有不同目标。判断 off-policy 应比较生成数据的行为与正在评价的策略，而不是观察是否有缓存。

<a id="lesson-check"></a>

## 10 · 带答案练习

问：回报 1、3，权重 .4、1.6，两种 IS 是多少？答：加权和 5.2，样本数和权重和恰为 2，因此均为 2.6。这次相等不是一般性质。

问：目标选 A，行为从不选 A，能否对缺失数据赋无限权重？答：不能。缺少支持意味着恒等式不成立，应改变采样或限定可评价目标。

问：同一 episode 中三次访问能当作三个独立 episode 做 bootstrap 吗？答：不能据此宣称独立。应按独立采样单位保留经历内相关结构。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](../../examples/tabular_textbook_lab.py)

```sh
python3 examples/tabular_textbook_lab.py monte-carlo
python3 examples/tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 预测的量是什么，误差又是什么？

先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

函数逼近与深度方法：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

持续学习中的研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](temporal-difference.md) → [投影与半梯度](../approximation/prediction.md) → [神经价值更新](../deep/deep-value.md) → [GVF 的问题与答案](../../textbook/gvf.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [价值预测与资格迹](../../textbook/value.md)
- [通用价值函数与预测知识](../../textbook/gvf.md)
- [知识保留与再适应](../../textbook/retention.md)

对应原始材料：5.1–5.7；5.9。本文为原创讲解，原书、论文与上游代码保留各自许可。
