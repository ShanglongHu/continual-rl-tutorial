# 策略梯度：从轨迹概率到 GAE 与 actor–critic

不对环境求导，怎样从采样动作计算策略梯度？价值网络和优势各自做什么？

## 本章内容

- 推导 score-function 梯度与 baseline 消去。
- 区分真实目标、优势估计和实现 surrogate。
- 给 GAE 设置独立的 bootstrap 与跨序列 mask。
- 正确使用 log_prob、detach 和 actor/critic loss。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

从当前随机策略产生的轨迹估计收益梯度，并用价值网络减少对完整回报的依赖。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 可微策略 $\pi_\theta$、价值近似 $V_\phi$、优势估计 $\hat A_t$。

### 需要求解的对象

构造与声明收益目标对应的 actor 更新，同时学习用于该估计器的 critic。

### 信息与数据权限

标准 on-policy 更新使用采样时策略的动作概率；使用旧策略数据需额外校正。

$$
\nabla_\theta J=\mathbb E\!\left[\sum_{t\ge0}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right]
$$

$A^\pi=Q^\pi-V^\pi$ 是真实优势。代码将它替换为估计量时，会引入方差以及可能的自举或截断偏差。

### 成立条件与解的含义

- 轨迹微分与期望交换合法；策略参数影响环境仅通过所执行动作。
- GAE 的有限 rollout 边界需正确区分真实终止和采样截断。

判断准则：先验证 score、优势与参数冻结时序，再将梯度估计的偏差方差与收益变化区分。

### 适用边界

- 由 GAE 降方差推断估计必然无偏或策略必然改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [策略梯度与 actor–critic](../../textbook/policy.md)：神经网络近似策略和价值，不改变 score-function 的概率学起点。

- 组合不同学习问题 · [时间信用分配与资格迹](../../textbook/credit.md)：GAE 在 critic 残差上分配时间权重，不等于完整递归网络参数信用。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

真实优势未知，长轨迹回报噪声大，截断窗口又留下未观察未来。

### 本章的核心思路

从精确策略梯度出发，把未知优势换成具有明确边界的多步残差估计。

1. [先求轨迹梯度](policy-gradient.md#lesson-derive)：环境不可微并不阻止对策略概率求导。

2. [构造时间上的优势估计](policy-gradient.md#lesson-advantage)：GAE 混合 TD 残差，并保留末状态的 bootstrap。

3. [隔离 actor 与 critic 的梯度职责](policy-gradient.md#lesson-algorithm)：actor 中通常停止对优势反传；critic 使用自己的回归损失。

结论与条件：真实优势给出梯度恒等式；实际 actor–critic 的性质还取决于 critic 误差和数据协议。

### 相关方法改变了什么

- REINFORCE / GAE：分别依赖完整采样回报和带自举的多步残差。

- VPG / A2C：共享策略梯度基础，但采样长度、critic 和同步方式可不同。


<a id="lesson-setting"></a>

## 1 · 随机策略和有限轨迹

价值方法通过比较动作价值来改变行为。连续动作难以逐个枚举，有限表示下的最好策略也可能需要特定的随机概率，这时可以直接学习策略。我们仍希望提高环境回报，但可调对象变为动作分布的参数；价值预测在这一过程中承担评价与减小噪声的作用。

策略 $\pi_\theta(a\mid s)$ 输出动作分布。离散动作通常使用 softmax logits，连续动作可使用 Gaussian。这里先考虑终止时间为 $T$ 的 episode，目标 $J(\theta)=\mathbb E_\theta[\sum_{t=0}^{T-1}\gamma^tR_{t+1}]$。环境转移对参数未知，但轨迹中动作的概率可以计算。

Actor 改变行为分布。学到的状态价值可以只充当 baseline，也可以进入 bootstrap 来评价动作。按教材 §13.5 的严格用法，REINFORCE 加一个学习的 baseline 仍是 Monte Carlo 策略梯度；后继价值参与动作评价时才构成这里的 actor–critic。现代工程有时用更宽的名称，阅读时以实际目标为准。A2C 通常同步收集并更新，A3C 使用异步工作者；执行方式不改变这个区分。

<a id="course-policy-class"></a>

## 1.1 · 先定义策略类：为何有时必须学习随机策略

有限折扣 MDP 在常规条件下存在确定性的平稳最优策略。这不表示观察受限、记忆受限或参数共享之后，受限策略类的最优解仍然确定。策略梯度的一个动机，是直接优化可表达的动作概率，而非只在固定 ε-greedy 规则下改变动作排序。

考虑 Sutton 与 Barto 的短走廊。非终止位置为零、一、二，终点在三。动作“右”在位置零、二向右，在位置一反而向左；“左”反向，位置零向左时原地不动。每步奖励 −1，本例使用无折扣的回合总回报（$\gamma=1$）。观察或参数化使三个位置使用相同概率 p 选择“右”。这里暂不允许 recurrent 记忆。

$$
\begin{aligned}
T_0&=1+(1-p)T_0+pT_1,\\
T_1&=1+pT_0+(1-p)T_2,\\
T_2&=1+(1-p)T_1.
\end{aligned}
$$

T 是从对应位置到达终点的期望步数。每式中的一，是刚刚执行的那一步。只有 0<p<1 时，起点的期望到达时间有限。

$$
J(p)=-T_0=-\frac{2(2-p)}{p(1-p)},\qquad
\frac{dT_0}{dp}=\frac{2(-2+4p-p^2)}{p^2(1-p)^2},\qquad
p^*=2-\sqrt2
$$

在可行区间中令导数为零，得到 p≈0.5858，期望回报约 −11.6569。p 为零会永远停在起点；p 为一会在前两个位置间循环。

复查策略类内的最优值；标准库即可执行。

```python
from math import sqrt
p = 2 - sqrt(2)
steps = 2 * (2 - p) / (p * (1 - p))
assert abs(steps - (6 + 4 * sqrt(2))) < 1e-10
print(p, -steps)  # 解析式核验，不是训练结果
```

若把真正的位置提供给策略，确定地依次选择右、左、右，只需三步。因此随机最优不是这个世界不可避免的性质，而是信息和策略类约束的结果。增加记忆、改善 agent state 和直接学习随机策略，是三个不同改动，实验应分开。

思考：如果动作概率已经接近零，softmax 梯度会很小，相关动作也很少被采到。新的任务要求反转这个偏好时，慢适应可能同时来自参数饱和与数据缺失。不能只增加 critic 容量，就断言问题已经解决。

<a id="lesson-derive"></a>

## 2 · 对轨迹概率求导，不对环境求导

$$
\begin{gathered}p_\theta(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1},r_{t+1}\mid s_t,a_t)\\\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\end{gathered}
$$

环境概率不含策略参数，所以对数导数只保留动作概率项。需要可微策略及使交换期望与求导成立的常规条件。

$$
\nabla J=\mathbb E\!\left[\sum_t\gamma^t\nabla\log\pi_\theta(A_t\mid S_t)G_t\right],\quad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}
$$

从完整 return 的 score estimator 出发，动作无法改变它之前已经发生的奖励，条件期望使这些过去奖励项为零，于是得到 reward-to-go。

$$
\sum_a\pi_\theta(a\mid s)\nabla\log\pi_\theta(a\mid s)b(s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0
$$

任何不依赖当前动作的 baseline 都可以在真实期望中消去。令 baseline 近似状态价值，便得到优势形式。baseline 自身参与训练，但 actor 的这次 score-function 求导将优势视为固定权重。

这里的“不依赖”是采样分布中的条件独立，不只是网络没有 action 输入。若先用当前样本的回报拟合 baseline，再评价同一个样本，拟合结果仍可能依赖它的动作。detach 只阻断计算图，不能消除这种统计依赖。可以在采样动作前固定 critic，或用独立 episode 训练基线；交叉拟合也应按独立完整轨迹分折。同一轨迹内随机拆分转移不能保证独立，因为后续样本仍可能透露当前动作的后果。

$$
L_\pi(\theta)=-\frac1n\sum_{i=1}^{n}\sum_{t=0}^{T_i-1}\gamma^t\log\pi_\theta(a_{i,t}\mid s_{i,t})\operatorname{sg}(\hat A_{i,t})
$$

这里 n 是独立 episode 数。采样策略处、优势满足相应无偏条件时，负损失梯度估计起点目标的梯度；它不是对任意远处候选策略都精确的回报函数。

$\gamma^t$ 对应从初始分布出发的折扣目标。很多实现均匀采样时间步并省去它，使用的是常见近似，或另一个状态加权目标；以随机的总步数归一化也不等于以固定 episode 数取平均。配套 PPO 小任务取 $\gamma=1$，按固定 rollout 步数训练，其实现 surrogate 与上述精确 episode 估计应分别理解。

<a id="lesson-advantage"></a>

## 3 · TD 残差、多步优势与 GAE

$$
\begin{gathered}\delta_t=R_{t+1}+\gamma b_tV_\phi(S_{t+1})-V_\phi(S_t)\\\hat A_t^{\mathrm{GAE}}=\delta_t+\gamma\lambda c_t\hat A_{t+1}^{\mathrm{GAE}}\end{gathered}
$$

$b_t$ 是能否 bootstrap；$c_t$ 是能否把下一行样本的优势继续接上。真正终止使二者为零。环境被人工重置的 timeout 通常 $b_t=1$、$c_t=0$。普通 batch 尾部用最后观测的价值，递推 carry 初始化为零。

![GAE 中价值自举与下一行优势的两条接续路径，在三种边界下分别开关](https://yingwen.io/crl-figures/concept-depth-classic-gae-masks.svg)

三栏固定奖励、旧价值与折扣参数，只比较不同边界语义。蓝箭头从最后观测取 V′；橙箭头接下一行的优势，只有它属于同一轨迹时才可接入。人工重置后不能使用重置观察代替最后观测。数值是 [GAE §3](https://arxiv.org/pdf/1506.02438#page=4) 递推的原创算例；时间截断的自举语义见 [Pardo 等 §3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf#page=5)。[计算代码](https://yingwen.io/crl-code/figures/classic-visual-depth.mjs)。

若序列内部没有边界，展开得到 $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$。$\lambda=0$ 只保留一步误差；$\lambda=1$ 在真终止 episode 中望远镜消去为 $G_t-V_\phi(S_t)$。不准确的 critic 在较小 $\lambda$ 下通常引入更多 bootstrap 偏差，较长估计又通常承受更多采样噪声。

$$
\begin{gathered}A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=\sum_{l=0}^{k-1}\gamma^lR_{t+l+1}+\gamma^kV_\phi(S_{t+k})-V_\phi(S_t)\\\hat A_t^{\rm GAE}=(1-\lambda)\sum_{k=1}^{K-1}\lambda^{k-1}A_t^{(k)}+\lambda^{K-1}A_t^{(K)}\end{gathered}
$$

在没有跨序列边界的 K 步片段上，先由 TD 残差望远镜相消得到 k 步优势，再混合不同长度。最后一项保留全部剩余权重，故权重和为一；真终止的尾值取零。K=1 时只有一步，λ=1 时只有 K 步估计，片段截断时仍含尾值。

第 $l$ 个残差出现在所有 $k\ge l+1$ 的项中。它的混合权重为 $(1-\lambda)\sum_{k=l+1}^{K-1}\lambda^{k-1}+\lambda^{K-1}=\lambda^l$，再乘 $\gamma^l$，就得到 GAE 的 $(\gamma\lambda)^l$。这解释了递推系数的来源，也避免把有限片段末端的余重丢掉。

如果 critic 恰为当前固定策略的真实价值、状态充分且边界正确，一步 TD 残差的条件均值已经是 $A^\pi(s,a)$；后续残差在按该策略采样的动作上均值为零，任意 $\lambda$ 的这种 GAE 都不会仅因 bootstrap 引入偏差。反过来，critic 即使用 Monte Carlo 训练，只要其动作评价仍不准确，也能使 actor 的更新有偏。原 GAE 论文的 $\gamma$-just 性质针对声明的折扣梯度估计，不能解读为对所有回报目标无偏。

先用旧 critic 计算全部优势和回归目标 $\hat R_t=\hat A_t+V_{\phi_{\rm old}}(s_t)$，再更新网络。优势标准化只能用于 actor 的权重；若把标准化后的优势加回 value 当作 critic target，就改变了价值的奖励单位。

<a id="course-gae-error"></a>

## 3.1 · GAE 的偏差到底从哪一项进入

固定采样策略 $\pi$，并在整段轨迹内固定 critic。记 $\epsilon_t=V_\phi(S_t)-v^\pi(S_t)$。用真价值形成的 TD 残差记为 $\delta_t^*$。先在长度 K、没有中间 reset 的同一段数据上比较两个估计，而不是把网络误差与轨迹噪声混在一起。

$$
\delta_t-\delta_t^*=\gamma\epsilon_{t+1}-\epsilon_t
$$

这是逐轨迹的代数恒等式，不需要平均。它说明一个价值误差会同时进入当前减项和前一步的自举项。

$$
\begin{aligned}
\hat A_t-\hat A_t^*&=\sum_{l=0}^{K-1}(\gamma\lambda)^l(\gamma\epsilon_{t+l+1}-\epsilon_{t+l})\\
&=-\epsilon_t+\gamma(1-\lambda)\sum_{l=1}^{K-1}(\gamma\lambda)^{l-1}\epsilon_{t+l}
+\gamma(\gamma\lambda)^{K-1}\epsilon_{t+K}.
\end{aligned}
$$

将相邻误差的系数合并。三项依次是起点 baseline 误差、中间 bootstrap 误差、有限片段的尾值误差。有限片段的优势递推从尾部的零开始；这不意味着尾部状态的价值为零。

在 actor 的条件期望里，$-\epsilon_t$ 与当前动作无关，可作为 baseline 消去。中间与尾部状态却依赖此前动作，不能同样删除。$\lambda=1$ 消掉中间项；只有真正终止且终止价值固定为零时，尾项才消失。无终止的采样窗口不会获得这个额外条件。

数值核验：$K=2,\gamma=0.9$，起点和终点误差为零，中间误差为二。残差误差分别为 $1.8,-2$。$\lambda=0$ 的优势误差为 1.8；$\lambda=0.8$ 时为 $1.8-0.72\times2=0.36$；$\lambda=1$ 时为零。这是 critic 误差传播的解析示例，不是“λ 越大越好”的实验结论。

即使 critic 恰是真价值，采样优势仍然有方差。更长的回报引入更多随机奖励和动作。真实任务中还要考虑策略是否在片段内变化、截断是否正确、价值目标是否共享参数，以及批次优势归一化对有限样本更新的影响。

$$
\mathbb E\!\left[\frac1B\sum_{i=1}^{B}\nabla\log\pi(A_i)(R_i-\bar R)\right]=\left(1-\frac1B\right)\nabla J,\qquad\bar R=\frac1B\sum_iR_i
$$

一个可精确检查的统计依赖：B 个独立同分布的单步 bandit 样本，均减去含自身回报的样本均值。交叉样本项的 score 期望为零，自身项则留下 1/B 的缩减。

B=1 时，这种中心化把学习信号完全消掉。用其余样本的均值作 leave-one-out baseline，可以在这个独立 bandit 例子中消除缩减。整段轨迹中的时间步并不独立，除以样本标准差又增加了随机缩放，因此不能把这条简单修正式直接用于所有优势标准化。

本书的逐样本即时更新协议不等待整批优势反向计算。资格迹可以把部分时间权重变成前向递推的记忆，但网络在递推期间也会改变。固定权重下的前向—后向等价，不自动等于非线性、逐步更新时的同一算法。后续在线信用分配章正是继续处理这个差别。

<a id="lesson-algorithm"></a>

## 4 · VPG / A2C 的更新次序与 autograd

**算法：这里写 γ=1 的有限时域形式；若采用初始状态折扣目标，actor 项还需对应时间权重。**

1. 固定本轮策略与 critic，收集新 trajectories。
1. 记录 observation、action、reward、terminal、value 与 next value。
1. 后向计算原始 GAE 与固定 critic target。
1. 构造 $L_\pi=-\operatorname{mean}(\log\pi_\theta(a\mid s)\operatorname{sg}(\hat A))$。
1. 构造 $L_V=\operatorname{mean}((V_\phi(s)-\operatorname{sg}(\hat R))^2)$。
1. 清空梯度，反传相应 loss，更新对应参数。
1. 丢弃本轮 on-policy 数据，开始下一轮采样。

离散动作必须对实际采样动作计算 `log_prob`，不能直接对最大 logit 求导。`torch.distributions.Categorical(logits=...)` 会进行稳定归一化。连续多维独立 Gaussian 的 `log_prob` 通常先返回各坐标，必须对动作维度求和，而不是把动作维度误当 batch。

如果共享 encoder，actor 和 critic 两个损失都会训练共享参数。必须明确损失系数与优化次序。把 critic 当 baseline 消去，并不意味着共享 critic 参数的梯度可以偷偷流过 actor 优势；这会增加与策略梯度不同的项。

<a id="experiment-deep-vpg"></a>

### 实验：实验 · 完整回报与一步 critic 提供不同的策略学习信号

策略梯度的目标相同，等待回报与借助 critic 会怎样改变有限预算学习？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步；actor 为 6→32 tanh→2，critic 为 6→32 tanh→1；PyTorch 默认随机初始化，Adam 的 actor 步长 0.003、critic 步长 0.01。VPG 每个完整回合更新一次；A2C 每 60 步用收集到的一步 TD 目标更新一次。γ=1。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** vpg.py 用完整 reward-to-go 减旧 critic 得到优势；a2c.py 用一步自举 target 减旧 critic。两者策略更新都停止优势的梯度。VPG 未完成的末尾回合不参与更新。

**测量。** 用隔离环境的贪心策略回报评价，而不是训练中的随机策略回报。还需读取 updated_batches：相同环境步数并没有固定 optimizer 更新次数。

```bash
python3 implementations/deep/vpg.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-vpg/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 environment_steps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 第 600 步 VPG 均值为 0.94，A2C 为 0.704，后者 seed 标准差约 0.528；第 1200 步两者都为 0.94。本配置说明 critic 引入后不必更早学好，但两者最终都达到最短路径行为。

**结论边界。** 更新频率和监督目标同时不同，不能把全部差异只归因于 bootstrap。小环境上的终点打平也不意味着估计器方差、随机策略和计算开销相同。

**继续实验。** 冻结一组完整轨迹，先比较两种优势与梯度，再做新的独立交互比较。固定样本导数正确与重新采样后表现更好，是两个命题。

[源码](../../implementations/deep/vpg.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-vpg/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-vpg/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-vpg/curves.json)

<a id="lesson-example"></a>

## 5 · 两步 GAE 与 softmax 梯度

取奖励 $(1,2)$，旧价值 $(0.5,1)$，第二步真终止，$\gamma=0.9,\lambda=0.8$。两步残差分别为 $1+0.9\times1-0.5=1.4$ 与 $2-1=1$。故优势为 $(2.12,1)$，critic 目标为 $(2.62,2)$。若第一步其实是独立序列的截断，不能把第二条的残差接到它后面。

$$
\frac{\partial\log\pi(a\mid s)}{\partial z_j}=\mathbf1[j=a]-\pi(j\mid s)
$$

softmax 的对数导数包含所有动作，不只是被选动作。若概率为 $(0.25,0.75)$，动作零的优势为 $2$，梯度上升方向为 $(1.5,-1.5)$，归一化会共同改变两个概率。

固定旧 value 和 next value，分别处理两类 mask，并输出原始优势与 critic target。

```python
def gae(rewards, values, next_values, terminated, boundaries,
        gamma=0.99, lam=0.95):
    n = len(rewards)
    if not all(len(x) == n for x in
               (values, next_values, terminated, boundaries)):
        raise ValueError('one next value and two masks per transition')
    advantages, carry = [0.0] * n, 0.0
    for t in reversed(range(n)):
        bootstrap, trace = masks(terminated[t], boundaries[t])
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * lam * trace * carry
        advantages[t] = carry
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def score_gradient(probabilities, action, advantage):
    # Derivative of A * log softmax(logits)[action] with fixed A.
    return [advantage * (float(i == action) - p)
            for i, p in enumerate(probabilities)]
```

<a id="lesson-code"></a>

## 6 · 从数值核验到实际 on-policy 训练

VPG 的一次 actor–critic 更新。优势和回归目标停止梯度；actor 更新后必须重新采样。

```python
def vpg_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               advantages, returns):
    # One update with fresh on-policy data; gamma=1 episodic convention.
    distribution = Categorical(logits=actor(x))
    loss_actor = -(distribution.log_prob(actions)*advantages.detach()).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()
    prediction = critic(x).squeeze(-1)
    assert prediction.shape == returns.shape
    loss_critic = ((prediction-returns.detach())**2).mean()
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    return float(loss_actor.detach()), float(loss_critic.detach())
```

PPO 使用同一套 actor–critic 采样和 GAE；区别在下一章的策略 surrogate。

```python
def train_ppo(epochs=20, seed=0, batch_steps=128):
    torch.manual_seed(seed)
    actor, critic = mlp(6, 2), mlp(6, 1)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.003)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=.01)
    env, initial_score = DeadlineChain(), evaluate(actor)
    observation = env.reset()
    log = {}
    for _ in range(epochs):
        xs, actions, logps, rewards, values, next_values, terminals = [], [], [], [], [], [], []
        for _ in range(batch_steps):
            with torch.no_grad():
                distribution = Categorical(logits=actor(observation))
                action = distribution.sample()
                logp, value = distribution.log_prob(action), float(critic(observation).item())
            xp, reward, terminal = env.step(int(action))
            with torch.no_grad():
                next_value = float(critic(xp).item())
            xs.append(observation); actions.append(action); logps.append(logp)
            rewards.append(reward); values.append(value); next_values.append(next_value); terminals.append(terminal)
            observation = env.reset() if terminal else xp
        # gamma=1 matches this finite-horizon, undiscounted task objective.
        advantages, returns = gae(rewards, values, next_values, terminals, terminals, gamma=1., lam=.95)
        log = ppo_update(actor, critic, actor_opt, critic_opt, torch.stack(xs),
                         torch.stack(actions), torch.stack(logps),
                         torch.tensor(advantages), torch.tensor(returns))
    return {'algorithm': 'PPO-Clip', 'seed': seed, 'environment_steps': epochs*batch_steps,
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(actor),
            **log, 'scope': 'DeadlineChain only; full-batch updates'}
```

执行 python3 examples/deep_textbook_lab.py demo 查看 GAE 与 score-gradient 的手算值；执行 python3 examples/deep_textbook_train.py ppo --epochs 20 --seed 0 查看真实 on-policy 采样。代码在 rollout 内固定参数，并保存旧 log-probability。环境已经真终止后才 reset；若 batch 恰在 episode 中间结束，环境活动继续保留。

VPG 更新核用固定优势乘当前 log_prob，进行一次 actor 更新，再要求重新采样；PPO 则使用旧策略概率比并对同批数据做受限复用。上面的完整采样循环运行 PPO，不是独立 VPG 训练命令。官方 VPG 文件提供独立、完整的工程入口；配套 VPG 核用于逐项检查梯度与 detach。

<a id="lesson-branches"></a>

## 7 · 估计误差与持续学习接口

- 优势偏差：critic 有误差、λ 小、轨迹截断都会改变估计。不能把所有 GAE 都称作无偏真实优势。
- 概率错误：离散采样后重新计算另一动作的 log_prob，或连续动作裁剪后仍使用裁剪前 Gaussian 密度，都改变了 estimator。
- 策略陈旧：旧轨迹反复用于裸 log-probability loss，不再是当前 on-policy 梯度。
- 任务变化：critic、状态和策略可以不同速度适应，优势的符号可能暂时错误。

在 CRL 中，策略熵变小可能让有用数据不再出现；critic 误差又会影响行动更新。诊断时应分别测覆盖、优势误差和网络学习能力，而不是看到回报下降就统一归因于遗忘。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 问：可以把即时奖励换成任意 baseline 吗？答：baseline 必须在条件期望中不依赖当前采样动作，且 actor 求导时固定它；否则消去推导不成立。
- 问：critic 的 loss 下降是否保证策略变好？答：不保证。它可能只改善高频无关状态，或在关键动作上的优势排序仍错误。
- 问：原始优势是 (2.12,1)，标准化后还能作为价值 target 吗？答：不能。actor 的尺度处理不应改变 critic 的奖励单位。
- 实验：固定一批完整轨迹，比较 λ=0、0.8、1 的目标，并单独制造一个 timeout。确认目标差异由哪一条 mask 和哪个尾值产生。



<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](../../examples/deep_textbook_lab.py)

```sh
python3 examples/deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Sutton et al. · Policy Gradient Methods with Function Approximation](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度定理与函数逼近的原始研究。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：GAE 的偏差、方差与 γ-just 估计条件。

- [Spinning Up · vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py)：官方教学工程；追踪 buffer 的 finish_path 与 actor/critic loss。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 怎样把较晚的反馈归给较早的计算？

多步回报定义用多远的未来构造目标。资格迹压缩过去的特征或梯度方向。前向与后向等价必须说明参数是在整个轨迹内固定，还是每一步改变。

函数逼近与深度方法：神经网络改变后，旧梯度不再等于用当前参数重算的梯度。递归状态还带来参数经过历史状态影响当前输出的路径，不能用一条普通 TD trace 代替。

持续学习中的研究问题：在每步计算有界的条件下，保留多少过去影响才有用？替换特征时，怎样处理与旧特征绑定的资格迹、优化器动量和元梯度？

[多步回报](../tabular/multistep.md) → [资格迹与等价条件](../approximation/traces.md) → [GAE 与 actor–critic](policy-gradient.md) → [在线信用分配](../../textbook/credit.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [策略梯度与 actor–critic](../../textbook/policy.md)
- [时间信用分配与资格迹](../../textbook/credit.md)
- [智能体状态与递归学习](../../textbook/state.md)

对应原始材料：Sutton & Barto §13.1–13.5；§12：资格迹与多步估计。本文为原创讲解，原书、论文与上游代码保留各自许可。
