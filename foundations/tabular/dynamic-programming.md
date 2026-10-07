# 动态规划：评价、改善与最优递推

已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。

## 本章内容

- 推导迭代评价、收缩与残差误差界。
- 证明策略改善，区分策略迭代、价值迭代和 GPI。
- 对照解析求解，理解计算预算与模型误差。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

已知有限 MDP 的完整模型。困难来自长期递推的计算，而不是从样本识别未知环境。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 模型核 $p$ 可精确查询并对所有后继状态求和；误差容限 $\varepsilon>0$。

### 需要求解的对象

计算固定策略的价值，或得到在该模型中近似最优的策略。

### 信息与数据权限

允许访问全部状态、动作与模型概率；这比只有一条交互轨迹的信息条件更强。

$$
v_\pi=T_\pi v_\pi,\qquad v_*=T_*v_*,\qquad \|v-T_*v\|_\infty\le\varepsilon
$$

$T_\pi$ 对下一动作按策略平均；$T_*$ 对动作取最大值。最后的残差是计算停止准则，不等于策略收益误差本身。

### 成立条件与解的含义

- 有限状态动作、精确平稳模型、有界奖励、折扣小于一。
- 异步更新需保证相关状态反复更新。

判断准则：残差、价值误差与提取策略的收益分别检查；使用收缩关系把残差转为可解释误差界。

### 适用边界

- 未知模型的样本复杂度或任意近似模型的真实性保证。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [价值预测与资格迹](../../textbook/value.md)：允许精确模型期望后，采样噪声从预测问题中消失。

- 组合不同学习问题 · [Dyna 与模型学习](../../textbook/dyna.md)：Dyna 学得模型后调用规划备份；模型误差仍需另外计算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

价值互相依赖，不能只按即时奖励贪心。

### 本章的核心思路

把互相依赖的方程视为收缩算子，并交替做策略评价和改善。

1. [建立可迭代的固定点](dynamic-programming.md#lesson-derive)：收缩性说明重复评价备份为何收敛，也给出停止误差。

2. [证明改善而不是凭直觉贪心](dynamic-programming.md#dp-improvement)：在旧策略价值下做一步贪心，再比较新策略的完整未来。

3. [选择评价与改善的计算比例](dynamic-programming.md#dp-algorithms)：策略迭代、价值迭代和 GPI 在同一关系中安排不同计算次序。

结论与条件：精确有限折扣条件下有评价收敛和策略改善结论；近似评价可破坏单调改善。

### 相关方法改变了什么

- 策略迭代 / 价值迭代：前者先较充分评价当前策略，后者将最优性备份直接混入每次更新。

- 同步 / 异步备份：改变计算调度；不能省去覆盖要求。


<a id="lesson-setting"></a>

## 1 · 已知模型下的计算问题

上一章的 Bellman 方程描述了价值应满足什么关系，却没有说明怎样求出价值。先假定每个动作的后果分布已知，就可以暂时不处理“怎样获得经验”，专心处理“怎样利用局部后果计算长期行为”。由此得到的策略评价与策略改善，也会成为后面从经验学习时的共同结构。

假设状态动作有限，模型 $p(s',r\mid s,a)$ 固定可查询，奖励有界且 $0\le\gamma<1$。算法能枚举动作的后果，不必真实执行动作才更新。一次 backup 指用后继价值形成当前估计目标的计算。

策略评价给定 π 求 vπ；策略改善用当前价值选择更好的动作。规划可以离线，也可以与交互并行。它与从真实经验直接学习的主要区别是后果来自模型，而不在于有没有使用名为 Q-learning 的更新。

本章使用同步扫描：先保留旧价值表，再计算新表。原地扫描允许立即使用本轮刚更新的值，它也可有良好性质，但数值过程与同步版不同。实现与推导应明确使用哪一种时序。

<a id="lesson-derive"></a>

## 2 · 迭代评价与收缩证明

$$
(T_\pi V)(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')],\qquad V_{k+1}=T_\pi V_k
$$

把固定点方程中未知的真实价值换成当前估计，得到迭代。真实终止后的 V 取零，但进入终点的奖励保留。

$$
|(T_\pi U)(s)-(T_\pi V)(s)|\le\gamma\sum_{a,s',r}\pi(a\mid s)p(s',r\mid s,a)\|U-V\|_\infty\le\gamma\|U-V\|_\infty
$$

奖励项相消；每个后继差不超过最大差；非负概率加权不会放大最大差。因此算子是收缩。

$$
\|V_k-v_\pi\|_\infty\le\gamma^k\|V_0-v_\pi\|_\infty,\qquad \|V-v_\pi\|_\infty\le\frac{\|T_\pi V-V\|_\infty}{1-\gamma}
$$

第一式来自反复应用收缩。第二式用三角不等式分解 V−vπ，再将含 γ 的误差项移到左侧。γ 接近一时，同样残差意味着更宽误差界。

<a id="dp-improvement"></a>

## 3 · 策略改善定理

假设已准确评价旧策略。构造新策略，使每个状态上“第一步用新选择、以后沿用旧策略”的价值都不差。需要证明一直使用新策略也不差，而不能把单步比较直接当作完整证明。

$$
T_{\pi'}v_\pi\ge v_\pi,\qquad v_\pi\le T_{\pi'}v_\pi\le T_{\pi'}^2v_\pi\le\cdots\longrightarrow v_{\pi'}
$$

第一项为假设。算子保序，所以可重复作用于不等式；收缩性保证极限为新策略的价值。因此逐状态得到 vπ′≥vπ。

$$
\pi'(s)\in\arg\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_\pi(s')]
$$

贪心改善满足单步不差条件。若改善后价值不再改变，就满足最优 Bellman 方程；唯一固定点意味着达到最优。

证明依赖准确评价和逐状态改善。近似价值可能错排动作；神经网络梯度可能改善一个状态而损害另一个状态。多个动作并列时采用固定规则或保留原动作，以免只因等价选择不断轮换而无法判停。

有限个确定性策略与单调改善共同解释策略迭代的停止。只要策略变化带来严格价值改善，就不会再次回到已严格劣于当前的策略。实现中的有限评价误差需要额外容差判断；若只在小数最后一位上交替改变并列动作，不能据此解释为环境的最优策略反复改变。

<a id="dp-algorithms"></a>

## 4 · 两种迭代与 GPI

**算法：策略迭代**

1. 初始化策略。
1. 重复：
  1. 固定策略，迭代评价到指定精度。
  1. 在每个状态枚举动作，以当前价值做一步贪心改善。
  1. 若策略完全不变，返回；否则继续评价新策略。

$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V_k(s')]
$$

价值迭代不等待某个策略评价结束，每轮直接做最优 backup。最优算子的收缩还用到两组最大值之差不超过逐动作最大差。

广义策略迭代 generalized policy iteration（GPI）概括评价与改善相互作用。部分评价后改善、逐状态交错都属于这一思想。它不是任何近似更新都会收敛的定理，也不同于迁移学习中同缩写的 generalized policy improvement。

价值还没有完全算准时，贪心策略能有多好？记 $T_*$ 为最优 Bellman 算子，$\pi_V$ 对当前表 $V$ 作一步贪心，且 $\eta=\|V-v_*\|_\infty$。在当前近似上有 $T_{\pi_V}V=T_*V$；在两侧各换回真实最优值，最多分别引入 $\gamma\eta$ 的误差。再传播这项单步损失，得到策略误差界。

$$
\|v_*-v_{\pi_V}\|_\infty\le\frac{2\gamma}{1-\gamma}\|V-v_*\|_\infty\le\frac{2\gamma}{(1-\gamma)^2}\|T_*V-V\|_\infty.
$$

第一步使用两个算子的收缩界及新策略价值的固定点关系；第二步使用最优算子的残差误差界。有限折扣、准确模型以及对全部状态取最大误差是这里的条件。

这个界给停止计算的容差一个行为含义，但可能相当保守。它不把某个小批次的平均 TD loss 转成策略保证，也没有计算模型错误。当 γ 接近1时，小的数值残差仍可能对应较大的可保证性能差；具体任务中还可检查动作间的价值差是否已足以稳定选择。

<a id="experiment-value_iteration"></a>

### 实验：实验 · 每轮改善，还是先把旧策略评价到很精确？

为什么策略评价的误差正在下降，相对于最优价值的误差却可能先上升？

**环境与可用信息。** 已知六格链模型，进入终点奖励 1，其余 −0.01，γ=0.95。价值初始全零。最优策略持续向右；策略迭代的初始策略却始终向左，因此会无限承受每步的小额负奖励。

**设置。** 各运行 1200 个同步模型扫描。价值迭代每轮对五个状态枚举动作并取最大值。对照每轮只评价当前策略，直到最大相邻迭代差小于 0.0000000001 才贪心改善。五个 seed 的运行完全确定。

**检验的机制。** 精确评价错误策略也会得到正确的负价值。它越接近该策略的真值，反而可能离最优价值更远。价值迭代不等待旧策略被精确求解，就将动作选择与价值更新交替进行。

**测量。** 纵轴始终为相对于解析最优价值的最大绝对误差。每轮都算五次状态备份，但价值迭代枚举两个动作，策略评价只读当前动作；策略改善还需额外动作计算。相同扫描数不是相同运行时间。

```bash
python3 implementations/classic/value_iteration.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/value_iteration/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 model_sweeps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 价值迭代在第一个保存点（第 15 轮）已达到浮点精度。对照第 15 轮误差为 1.1073，第 300 轮接近 1.2，之后在第 600 轮前也降到浮点精度。早期上升来自正在评价初始向左策略，不能据此声称 Bellman 评价发散。

**结论边界。** 日志每 15 轮保存一次，不能从这张图读出价值迭代准确用了几轮。对照的极严格评价容差明显增加等待；它不代表所有策略迭代实现，更不是价值迭代普遍更快的证明。

**继续实验。** 在同一模型上，将每次改善之前的评价轮数设为 1、5 和直到容差满足，比较 modified policy iteration。记录总动作模型调用、策略变化次数与残差，分别解释目标误差和求解器误差。

[源码](../../implementations/classic/value_iteration.py) · [逐种子记录](https://yingwen.io/crl-code/results/value_iteration/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/value_iteration/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/value_iteration/curves.json)

<a id="lesson-example"></a>

## 5 · 完整手算

A 退出得 1 或继续零奖励到 B；B 退出得 2 或继续得 .5 回 A；γ=.9。初始都退出，价值为 (1,2)。A 继续值为 1.8，B 继续值为 1.4，所以第一轮仅 A 改变。

| 策略 | 评价方程 | 结果 |
| --- | --- | --- |
| 都退出 | vA=1，vB=2 | (1,2) |
| A继续、B退出 | vA=.9vB，vB=2 | (1.8,2) |
| 都继续 | vA=.9vB，vB=.5+.9vA | (45/19,50/19) |

第二轮 B 的继续值变成 .5+.9×1.8=2.12，才超过退出值。最终价值约 (2.368421,2.631579)，再次贪心保持不变。改进行为发生在不同轮次，说明即时收益比较不能代替长期评价。

<a id="lesson-code"></a>

## 6 · 求解器与检查

同步评价、固定并列选择、策略迭代和价值迭代。模型在完整脚本的 mdps 区域。

```python
def evaluate_policy(model, policy, gamma=0.9, tolerance=1e-12):
    if not 0 <= gamma < 1:
        raise ValueError("this discounted solver requires gamma < 1")
    values = {s: 0.0 for s in model}
    for _ in range(100000):
        new = {s: sum(p*q for p, q in zip(
            policy[s], action_values(model, s, values, gamma))) for s in model}
        difference = max(abs(new[s]-values[s]) for s in model)
        values = new
        if difference < tolerance:
            return values
    raise RuntimeError("evaluation did not converge")

def greedy_policy(model, values, gamma):
    policy = {}
    for s in model:
        q = action_values(model, s, values, gamma)
        best = max(range(len(q)), key=lambda a: q[a])
        policy[s] = [float(a == best) for a in range(len(q))]
    return policy

def policy_iteration(model=MODEL, gamma=0.9):
    policy = {s: [1.] + [0.]*(len(model[s])-1) for s in model}
    history = []
    while True:
        values = evaluate_policy(model, policy, gamma)
        history.append({"actions": [row.index(1.) for row in policy.values()],
                        "values": list(values.values())})
        improved = greedy_policy(model, values, gamma)
        if improved == policy:
            return values, policy, history
        policy = improved

def value_iteration(model=MODEL, gamma=0.9, tolerance=1e-12):
    if not 0 <= gamma < 1:
        raise ValueError("this discounted solver requires gamma < 1")
    values = {s: 0.0 for s in model}
    for sweep in range(100000):
        new = {s: max(action_values(model, s, values, gamma)) for s in model}
        difference = max(abs(new[s]-values[s]) for s in model)
        values = new
        if difference < tolerance:
            return values, greedy_policy(model, values, gamma), sweep+1
    raise RuntimeError("value iteration did not converge")
```

dynamic-programming 模式打印三轮策略和价值；默认逐轮差阈值 $10^{-12}$ 下，价值迭代为 246 轮，得到相同策略。阈值用于停止数值计算，不自动等于策略性能差；需要误差界时应另算 Bellman 残差。

求解器拒绝 γ=1。适当终止的无折扣问题可以建立其他收敛条件，但不能只删除这个检查就继续引用本章折扣收缩证明。测试同时检查解析价值、策略改善和收缩不等式。

价值迭代也有收缩性质。两个动作值向量的最大值之差，不超过各对应动作值之差的最大绝对值；再使用后继价值的概率加权界，得到与策略评价相同的 γ 因子。不同之处是每次 backup 同时重新选择贪心动作，而不是固定一个策略做平均。

实验可分别取 γ=.5、.9、.99，记录达到同一残差要求所需的扫描次数，并重新求最优策略。前者检验长期依赖给数值计算带来的困难，后者检验目标本身的变化。不能假定折扣变化只影响速度而不影响动作偏好。将同步扫描改成原地扫描后，应比较固定点与中间过程，而非要求每轮数值完全一样。

<a id="lesson-branches"></a>

## 7 · 计算分配与模型错误

完整扫描可能把预算花在变化很小的状态上。异步 DP 更新部分状态，通常还要求相关状态持续得到更新；优先扫描则根据估计变化安排计算。局部更新的调度、一次 backup 的成本和达到同等误差的总计算，应分别报告。

动态规划解的是输入模型。错误模型可以有一个精确但错误的固定点，增加迭代次数只能更准确地求解它。持续环境中的模型更新、失效检测和旧知识保留，需要与规划调度共同考虑。

状态很多时，即使模型精确，逐项枚举也可能不可行。后续函数逼近共享状态表示，采样规划减少枚举后果，时间抽象减少决策深度；它们改变不同资源瓶颈，不能笼统称为“更大的动态规划”。

<a id="lesson-check"></a>

## 8 · 带答案练习

问：逐轮差很小是否一定准确？答：还取决于 γ 与模型。残差误差界含 1/(1−γ)，模型偏差则不在该纯计算误差界内。

问：为什么能同时改变所有状态动作？答：准确评价时，所有状态满足同一个单步不差条件，算子单调性把它传播到整个未来。估计误差下不能直接套用。

问：评价一定从零开始吗？答：不需要，有限折扣情形可从任意有限表开始。初始化改变计算过程，不改变固定点。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](../../examples/tabular_textbook_lab.py)

```sh
python3 examples/tabular_textbook_lab.py dynamic-programming
python3 examples/tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](dynamic-programming.md) → [策略梯度定理](../approximation/policy-gradient.md) → [TRPO 与 PPO 的近似](../deep/trust-region.md) → [持续控制的比较器](../../textbook/control.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [价值预测与资格迹](../../textbook/value.md)
- [时间抽象与规划](../../textbook/planning.md)

对应原始材料：4.1–4.7。本文为原创讲解，原书、论文与上游代码保留各自许可。
