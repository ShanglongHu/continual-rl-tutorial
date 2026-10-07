# 对手建模与递归推理：预测谁，回应什么？

给定合作或竞争的评价目标，怎样利用他者模型和有限递归改善响应，并检验其是否真的有用？

## 本章内容

- 区分外层评价与目标构建、内层行为预测与响应学习。
- 理解自对弈、历史平均、FSP、NFSP 和反事实遗憾。
- 推导 PR2 的变分响应与 GR2 的有限递归，辨明理论和实现边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 随机博弈与信息集

联合动作决定转移；各参与者只能按自己的可用信息行动。

### 策略评价与改善

评价固定策略与求最优响应是不同问题。

### 概率、熵与 KL

区分真实分布与拟合模型；知道 KL 非负。

<a id="problem-definition"></a>

## 本章的问题定义

有限双人博弈，或具有已声明观察权限的折扣随机博弈。对手可以固定、从一个分布抽取，或随训练更新；三种协议分别讨论。

### 给定条件与符号

- 自身可用历史、动作集和奖励。
- 对手身份与已执行动作是否可见，是否有共同随机信号。
- 对手分布 q、模型族、更新预算和独立评价对手。

### 需要求解的对象

学习可检验的行为或响应模型，并求对指定对手分布的近似最优响应；均衡目标还需检查所有单方偏离。

### 信息与数据权限

同时行动时，不能把自身尚未执行的本步动作视为对手已收到的观察。训练期记录联合动作不等于执行前可见。

$$
J_i(\pi_i;q)=\mathbb E_{\pi_{-i}\sim q}\mathbb E_{\pi_i,\pi_{-i}}\!\left[\sum_{t=0}^{T-1}\gamma^tR^i_{t+1}\right],\quad \operatorname{BR}_i(q)\in\arg\max_{\pi_i}J_i(\pi_i;q)
$$

q 每局抽取完整对手策略，局内冻结。无限时域取 0≤γ<1、奖励有界。对固定 q 求响应不等于求 Nash 均衡。

### 成立条件与解的含义

- 策略遵守既定信息结构；历史摘要的充分性需独立论证。
- 矩阵算例使用已知收益；神经实验还存在采样和逼近误差。
- 精确响应、无遗憾和收敛结论保留其博弈类别与优化条件。

判断准则：分别检查预测误差、固定对手收益、交叉对战与偏离收益，不能彼此替代。

### 适用边界

- 不从条件相关推断因果影响。
- 不把内部递归层数当作真实对手的心理层次。
- 不声称深度 PR2、GR2、NFSP 在任意游戏中收敛。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [持续控制与学习智能体比较](../../textbook/control.md)：对手分布进入控制目标，单方策略改善与均衡评价需要分开。

- 组合不同学习问题 · [智能体状态与递归学习](../../textbook/state.md)：对手类型不可见时，历史和信念进入行为预测与控制。

- 组合不同学习问题 · [元学习与学习规则的适应](../../textbook/meta.md)：选择响应对象与更新规则发生在基础策略学习之外，不能与局内推理混同。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

对手行为、训练分布与双方更新相互影响，短期胜利可能只是利用暂时弱点。

### 本章的核心思路

先固定评价对象，再分开数据生成、模型推断与策略响应。

1. [明确对谁学习](multi-agent-reasoning.md#lesson-self-play)：区分最新对手和历史平均。

2. [定义模型](multi-agent-reasoning.md#lesson-opponent-model)：区别行为拟合、假设响应与真实信息权限。

3. [构造响应](multi-agent-reasoning.md#lesson-pr2)：从变分目标到有限层递归。

4. [检验边界](multi-agent-reasoning.md#lesson-limits)：分别评价模型、收益、预算和偏离。

结论与条件：解析恒等式可精确验证；它们不推出神经网络训练的普遍收敛。

### 相关方法改变了什么

- NFSP：用历史平均构造训练分布，不必显式预测候选动作的响应。

- CFR：在信息集上累计反事实遗憾，不拟合条件对手模型。

- PSRO：评价并扩充完整策略种群，不等于增加内部推理层数。


<a id="lesson-setting"></a>

## 1 · 他者建模在多智能体学习中的位置

本章讨论合作与开放式多智能体学习的支撑方法：预测其他参与者，并据此改善自己的响应。这里的“对手模型”也可用于伙伴；建模对象的名称不决定双方的奖励关系。

在[合作多智能体学习](multi-agent.md)中，核心问题包括怎样发现有效的联合行为，以及怎样把共同结果归因于各自行动。预测伙伴能帮助组织探索和控制，但不会自动解决联合探索或信用分配。在[开放式多智能体学习](multi-agent-populations.md)中，竞争与开放合作还需评价现有策略、构建下一轮对手或伙伴分布，再检查策略改善。更准确的模型可以提高内层响应质量，却不替代外层评价和目标构建，也不保证整体性能单调提高。

面对另一个学习者，先问评价时对手是谁，再问对其行为知道什么，最后问下一轮应训练什么策略。三个问题分别决定回报分布、预测模型与改善算子。

| 对象 | 输入与输出 | 边界 |
| --- | --- | --- |
| 对手分布 q | 给完整策略或历史版本分配概率 | 近期对手不是未来所有对手 |
| 行为模型 ρ | 给定可用信息预测对手行为 | 预测准确不自动使自身稳健 |
| 响应算子 BR | 给定对手分布和自身奖励，求改善策略 | 一次响应不是双方均衡 |

对手固定、全状态可见且各方使用 Markov 策略时，可以积分掉对手动作，得到单智能体 MDP。若对手依赖私有历史，或每局抽取未知类型，当前观察通常仍非 Markov。冻结对手消除了参数更新，不会消除隐藏信息。

以下先用矩阵游戏隔离学习动力学，再扩展到序列决策。随机博弈、局部信息和联合价值的共同设定见[合作多智能体学习](multi-agent.md)；这里在这些设定之上明确模型预测什么、响应优化什么。

<a id="lesson-self-play"></a>

## 2 · 自对弈也在产生训练问题

$$
\pi_i^{k+1}\approx\operatorname{BR}_i(q_{-i}^{k}),\qquad q_{-i}^{k}=\delta_{\pi_{-i}^{k}}
$$

最新策略自对弈用对手当前版本定义本轮目标。实际可只做少量梯度步，而非求完整 best response。

每轮固定训练对手，采集对局，再更新策略。轮换与同时训练有不同数据分布。共享网络是对称任务中的一种实现，不是自对弈的定义；不对称游戏仍可为不同角色维护不同策略。

石头—剪刀—布中，只回应最新纯策略会反复经历石头、布、剪刀。每次响应都正确，最新策略却始终容易被利用。“战胜上一轮”因此不等于全局进步。

$$
\operatorname{Gap}(x,y)=\max_a(Ay)_a-\min_b(x^\top A)_b
$$

A 是有限双人零和游戏的行玩家收益矩阵。gap 等于双方最优单方偏离收益之和；零 gap 才说明该策略对是均衡。

收益回答“对这一分布表现如何”，gap 回答“还有哪些偏离能获利”。只报告相邻训练版本的胜率，会漏掉循环和未遇见的克制策略。

<a id="lesson-fictitious"></a>

## 3 · 虚拟博弈：回应历史平均

$$
\beta_i^{k+1}\in\operatorname{BR}_i(\bar\pi_{-i}^{k}),\qquad \bar\pi_i^{k+1}=\frac{k\bar\pi_i^{k}+\beta_i^{k+1}}{k+1}
$$

有限正规形游戏中，平均完整策略的概率分布；k 是已纳入平均的响应数。

历史平均保留已经遇到的行为，避免只追逐最新对手。经典虚拟博弈在有限双人零和等游戏中有平均策略的收敛结论，不是一般和游戏或最新策略的普遍结论。

序列游戏不能简单地对每个信息集的动作概率做等权平均。几乎不到达某处的策略，不应与经常到达的策略在该处获得相同权重。Fictitious Self-Play 用实现概率处理这个问题。

$$
\bar\pi_i(a\mid I)=\frac{\sum_k w_k r_i^{\pi_i^k}(I)\pi_i^k(a\mid I)}{\sum_k w_k r_i^{\pi_i^k}(I)}
$$

I 是信息集，r 是玩家自身动作对到达 I 的概率贡献。完美回忆下，它与先按 w 抽取完整策略的混合实现等价；分母为零处可任意定义。网络参数平均不能替代此式。

两步手算：完整策略甲始终左，乙始终右，回合开始各半抽取，则路径 LL、RR 各有一半，LR、RL 不发生。若每一步都等权平均两种动作，四条路径各有四分之一。完美回忆的第二步信息集记住自己先前选了左还是右：到达左分支时甲的自身实现概率为一、乙为零，上式便在该分支只取甲的左动作；右分支反之。由此恢复完整策略混合的路径分布，而不是独立重抽。

<a id="lesson-nfsp"></a>

## 4 · NFSP：响应学习与平均策略学习

NFSP 用 Q 网络近似响应，用另一网络拟合自己的历史响应行为。前者学习怎样获胜，后者保留已经采用过什么。平均策略网络不是用来拟合对手动作的。

$$
\sigma_i=(1-\eta)\bar\pi_i+\eta\beta_i,\qquad 0<\eta<1
$$

完整策略层面的 anticipatory mixture；原算法在每局开始选择平均或响应模式，不是每一步重新抽模式。

$$
L_{\rm SL}(\vartheta)=\mathbb E_{(I,a)\sim\mathcal M_{\rm SL}}[-\log\bar\pi_\vartheta(a\mid I)]
$$

只将执行响应模式时的自身行为写入监督记忆。Reservoir sampling 使有限记忆近似保留整段行为流，不保证精确保存全部策略。

**算法：两种记忆的对象、采样方式和用途不同。**

1. 每局开始，以概率 η 执行近似响应 β，否则执行平均策略。
1. 所有转移写入 RL replay。
1. 仅将响应模式下的自身 (信息集, 动作) 写入 SL reservoir。
1. 用 Q-learning 更新响应；用交叉熵更新平均策略。
1. 冻结平均策略，独立计算或估计可利用性。

改用最近窗口会改变所拟合的平均对象。非线性逼近、有限记忆与不精确响应也会改变理论条件。NFSP 是 FSP 的可扩展近似，不是“给 DQN 加一个网络就保证 Nash”。

<a id="lesson-cfr"></a>

## 5 · CFR：从反事实遗憾组织自对弈

CFR 问：在自己能够决策的信息集，若换一个动作，累计会少后悔多少？它据此组织下一轮策略。COMA 的反事实 baseline 则服务于策略梯度的方差缩减；二者不是同一种更新。

$$
v_i^\sigma(I,a)=\sum_{h\in I}\rho_{-i}^\sigma(h)\sum_{z\succeq ha}\rho^\sigma(ha,z)u_i(z)
$$

h 是信息集内历史，z 是终局。第一个权重含对手与 chance 到达 h 的贡献，排除自身到达概率；之后强制选 a，再按 σ 继续。此值不是归一化的条件期望。

$$
\begin{aligned}r_i^k(I,a)&=v_i^{\sigma^k}(I,a)-\sum_b\sigma_i^k(b\mid I)v_i^{\sigma^k}(I,b),\\R_i^K(I,a)&=\sum_{k=1}^K r_i^k(I,a),\\\sigma_i^{K+1}(a\mid I)&=\frac{[R_i^K(I,a)]_+}{\sum_b[R_i^K(I,b)]_+}.\end{aligned}
$$

分母为零时取均匀分布。这是原始累计遗憾的 regret matching；CFR+ 的逐轮截断是另一更新。

有限、完美回忆的双人零和游戏中，局部反事实遗憾控制整体外部遗憾，再由双方平均遗憾界控制平均策略的均衡误差。采样、神经近似和游戏抽象各有额外条件；局部指标小不能独立证明任意深度策略达到均衡。

<a id="lesson-opponent-model"></a>

## 6 · 行为预测不等于真实因果响应

$$
L_{\rm pred}(\phi)=-\mathbb E_{(H_i,A_{-i})\sim D}\log\rho_\phi(A_{-i}\mid H_i)
$$

用允许观测的历史预测对手已执行动作。动作不可观察时需额外推断，不能凭空得到监督标签。

条件模型还可输入候选自身动作，写成 $\rho(a_{-i}\mid s,a_i)$。它可以表示数据相关性，也可以表示内部假设的响应；二者都不意味着同时行动的对手已经看到了 $a_i$。

$$
\pi(a_i,a_{-i}\mid s)=\pi_i(a_i\mid s)\pi_{-i}(a_{-i}\mid s)
$$

给定同一个完整状态、双方独立随机行动时，真实联合分布满足此因子化。隐变量、共同信号或混合的训练版本会改变条件独立关系。

要把条件相关解释为“我改变动作就使对手改变动作”，需实际观察顺序、承诺机制或因果假设。先行动并被对手看见的 Stackelberg 游戏，与同时行动的 Nash 游戏是不同问题。

$$
\widetilde Q_i(s,a_i)=\sum_b\rho_\phi(b\mid s,a_i)Q_i(s,a_i,b)
$$

固定 φ 和 Q 后，这是模型内的期望评分。真实评价仍需由环境中的实际对手执行。

actor 可能选择数据未覆盖的动作，利用模型虚构的有利回应。应分别检查保留轨迹上的预测误差、动作覆盖、模型失配以及环境真实回报。

<a id="lesson-pr2"></a>

## 7 · PR2：从价值构造变分响应

PR2 使用条件对手模型与概率推断近似响应，再改进自身策略。重点不是把对手网络拼到输入，而是规定响应分布怎样从价值和正则项产生。下面在有限动作上推导其软响应核。

$$
F_i(s,a_i)=\alpha\log\sum_b e^{Q_i(s,a_i,b)/\alpha},\qquad \rho^*(b\mid s,a_i)=e^{[Q_i(s,a_i,b)-F_i(s,a_i)]/\alpha}
$$

α>0。F 是软聚合值，不是固定真实对手下的普通价值。连续动作还需声明积分的基准测度与可积性。

$$
\alpha D_{\rm KL}(\rho\Vert\rho^*)=F_i-\mathbb E_\rho Q_i-\alpha H(\rho)
$$

代入 log ρ*=(Q−F)/α 并使用概率和为一，即得恒等式。

$$
F_i=\max_\rho\{\mathbb E_\rho Q_i+\alpha H(\rho)\}
$$

KL 非负给出上界，ρ=ρ* 时达到。限制模型族后，最小化 KL 是变分近似；有限粒子又增加采样近似。

指数中使用的是玩家 i 的价值，不能因此说对手在最大化其自身奖励。这是推断构造中的响应目标，不是任意竞争对手的通用行为定律。未经检验地替代真实对手分布，可能产生乐观失配。

取 Q=(0,1)、α=1，ρ*≈(0.269,0.731)，普通期望≈0.731，熵≈0.582，F≈1.313。F 大于最大 Q=1 不代表获得额外环境奖励；差额来自正则化。若真实对手总选第一项，期望仍为零。

$$
(\mathcal T^{\pi_i}Q_i)(s,a_i,b)=r_i(s,a_i,b)+\gamma\mathbb E_{s',a_i'\sim\pi_i}[F_i(s',a_i')]
$$

与上述软响应配套的理想固定自身策略备份。神经 critic 用停止梯度的 target 版本计算右端，再回归当前 joint Q；若任务真实终止则不 bootstrap。PR2 的软值不能换成普通对手期望而仍称同一算子。

先只检查这个固定策略算子。若 $\|Q_1-Q_2\|_\infty\le\varepsilon$，逐项比较指数可得 $e^{-\varepsilon/\alpha}\sum_b e^{Q_2/\alpha}\le\sum_b e^{Q_1/\alpha}\le e^{\varepsilon/\alpha}\sum_b e^{Q_2/\alpha}$。取对数并乘温度，便有 $|F_1-F_2|\le\varepsilon$；相同固定转移与自身策略的期望再乘折扣，使备份差不超过 $\gamma\varepsilon$。有限动作、有界奖励、$\gamma<1$ 下，这是模型内部固定备份的压缩性；不证明真实对手等于推断响应，也不证明双方同时改变策略和模型时达到均衡。原文针对自对弈的更强结论还声明了额外博弈条件。

<a id="lesson-response-gradient"></a>

## 8 · 回应模型时，对什么求导？

$$
\nabla_a\widetilde Q(a)=\sum_b\rho_\phi(b\mid a)\nabla_aQ(a,b)+\sum_bQ(a,b)\nabla_a\rho_\phi(b\mid a)
$$

固定 φ，对候选动作 a 求导。第一项改变自身动作，第二项计入模型预测的对手分布变化。

stop-gradient 对手响应会删除第二项，因此改变更新算子，而不只是节省计算。保留第二项也只说明优化了模型评分，不能把它当作真实对手的因果导数。

$$
\nabla_a F(a)=\sum_b\rho^*(b\mid a)\nabla_aQ(a,b)
$$

直接微分 log-sum-exp 即得。它不与上式冲突：F 还包含熵，ρ* 是内层最优解。不能混用普通期望与软值的梯度。

**算法：PR2/GR2 型工程的阅读顺序；不是宣称不同论文采用相同的软聚合算子。**

1. 声明对手协议、可用信息、响应模型和正则化目标。
1. 收集自身奖励、后继观察以及允许记录的联合动作。
1. 固定 target 版本，用选定软 Bellman 目标更新 joint critic。
1. 用 KL / 粒子近似更新条件响应模型。
1. 更新 actor，明确哪些响应分支 stop-gradient；更新 target。
1. 在冻结且未参与拟合的对手上独立评价。

PR2-Q 与 PR2-Actor-Critic 分别使用价值控制与 actor–critic。原算法的数据元组包含对手已执行动作；分散训练不等于“不需要任何其他玩家行为信息”。

<a id="lesson-gr2"></a>

## 9 · GR2：有限递归与层次混合

模型还可以假设：对手正在回应一个关于我的模型。递归必须从 level-0 行为假设开始，并在有限深度停止。零层可以是均匀行为或学习到的基础策略，不是未经定义的“不会思考”。

$$
\pi_i^{(k)}=\operatorname{BR}_i(\pi_{-i}^{(k-1)}),\qquad k\ge1
$$

单状态、精确响应的教学抽象。GR2-L 以条件策略交替展开自身与对手的推理；实际更新不逐层求精确 BR。

$$
q_{-i}^{(<k)}=\sum_{\ell=0}^{k-1}w_\ell^{(k)}\pi_{-i}^{(\ell)},\quad w_\ell^{(k)}=\frac{\lambda^\ell/\ell!}{\sum_{j=0}^{k-1}\lambda^j/j!},\quad \pi_i^{(k)}=\operatorname{BR}_i(q_{-i}^{(<k)})
$$

λ>0。截断 Poisson 混合说明 cognitive hierarchy 与 GR2-M 的动机；不声称作者工程逐行实现这个教学形式。

用猜数说明响应的精确含义。n 人选数，目标为全体均值的 p 倍。已知其他 n−1 人均值 m，自己命中目标需满足 x=p[x+(n−1)m]/n，故 x=p(n−1)m/(n−p)。常见 x≈pm 忽略自身对均值的影响，只在 n 很大等近似条件下成立。

取 n=2、p=0.7、零层 m=50，精确响应≈26.923，不是 35。再假设对手处于该一层，二层响应≈14.497。这是固定对手数字、平方偏差最小化的教学任务；胜负奖励、随机对手与并列规则需另定义响应。

GR2 实践使用确定性内部展开、跨层参数共享与辅助层间改善目标控制成本。深层也重复使用有偏模型，未必更准。作者工程还截断部分对手分支梯度；递归展开不等于所有层完整反向传播。

<a id="lesson-rommeo"></a>

## 10 · ROMMEO：有利响应不能脱离经验依据

PR2 之后的一个自然问题是：模型偏向有利行为时，怎样防止它想象一个现实中不存在的合作伙伴？Tian、Wen 等的 ROMMEO 从合作决策的概率推断出发，用观测到的行为分布约束对手模型。它不是简单地提高行为预测准确率，而是在协调收益与经验依据之间建立明确目标。

$$
\mathcal J_s(\pi,\rho)=\mathbb E_{b\sim\rho,a\sim\pi(\cdot\mid s,b)}Q(s,a,b)+\alpha\mathbb E_{b\sim\rho}H(\pi(\cdot\mid s,b))-D_{\rm KL}(\rho(\cdot\mid s)\Vert P(\cdot\mid s))
$$

固定状态的改善子问题。P 是经验对手先验；这里 KL 系数为一，与原文该形式对应。π 的条件输入 b 在执行时来自内部模型，不等于提前偷看真实动作。

$$
F(s,b)=\alpha\log\sum_a e^{Q(s,a,b)/\alpha},\qquad \rho^*(b\mid s)=\frac{P(b\mid s)e^{F(s,b)}}{\sum_{b'}P(b'\mid s)e^{F(s,b')}}
$$

先对自身条件策略优化，再对 ρ 优化，分别得到软响应与经验先验的指数倾斜。与 PR2 中对对手动作做无先验软聚合不同，ROMMEO 明确保留 P 的约束。

若 P(b|s)=0，有限 KL 不允许模型给该行为正概率。这个限制能抑制虚构响应，也可能排除尚未观察到的有益协调方式。怎样平滑先验和收集覆盖需要另行定义。原文的理论设置及实验重点是合作游戏；不能把其乐观协调模型当作零和对抗的安全保证。

<a id="lesson-gscu"></a>

## 11 · GSCU：何时利用模型，何时保守行动？

另一个问题不在递归深度，而在于是否应相信模型。Fu、Tian、Wen 等的 GSCU 先离线学习对手策略的连续嵌入，并训练一个条件化该嵌入的近似响应。线上根据已经完成的对局更新嵌入后验，再在实时利用策略与固定保守策略之间做 bandit 选择。

该设置允许对手改变，但准备阶段已经提供训练对手集合与较强的保守策略。原文主要设定还允许读取历史对局中对手的观察—动作轨迹；当前局的未知动作并未提前提供。后验分布表达模型内的不确定性，不保证涵盖训练外对手。

原算法将嵌入的后验均值输入条件响应策略。EXP3 根据获得的回报更新两个候选的选择权重；它不是检查后验方差是否超过某个阈值，再机械切换到保守策略。

$$
\max_{e\in\{\mathrm{greedy},\mathrm{safe}\}}\sum_{j=1}^{T}g_j(e)-\mathbb E\sum_{j=1}^{T}g_j(E_j)=O(\Delta\sqrt T)
$$

这是两臂 EXP3 型选择器的外部遗憾尺度；每局反馈有界于跨度 Δ，且需遵守相应 bandit 协议。比较对象是两个候选过程，不是所有可能策略或每局事后最优选择。

若对手会因自己历史选择而改变，“相同实现反馈序列上的外部遗憾”和“如果一直采取另一策略，对手本来会怎样反应”也不同。不能把前者直接称为长期因果安全。GSCU 说明有价值的分工：后验推断管理模型不确定性，条件策略产生响应，外层选择器控制使用哪个候选。它不要求线上重新训练整个响应网络。

<a id="lesson-limits"></a>

## 12 · 理论边界：稳定的是哪个过程？

PR2 软价值迭代的结论具有对称性、特定均衡和价值算子条件，不能外推到任意一般和游戏及非线性 actor–critic。GR2 的均衡存在性涉及构造的推理博弈，不等于学习必然找到它。

$$
\frac{d}{dt}\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0&a\\b&0\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix},\qquad ab<0
$$

两动作单状态游戏内部均衡附近的一类连续时间动力学；特征值 ±i√(−ab) 表示绕行。这种中性稳定不应简单叫作发散。

$$
M_\zeta=\begin{bmatrix}\zeta ab&a\\b&\zeta ab\end{bmatrix},\qquad \lambda(M_\zeta)=\zeta ab\pm i\sqrt{-ab}
$$

在该局部模型中加入 ζ>0 的预期响应项，实部变负。它说明某种预期更新可阻尼旋转，不证明任意递归深度都更好。

$$
0<\eta<\frac{2\zeta}{1-\zeta^2ab}
$$

显式 Euler 更新还需此步长条件，使 |1+ηλ|<1。连续时间稳定不保证任意学习率下稳定；此模型不覆盖概率边界与神经随机梯度。

- 预测检验：新轨迹上的概率校准与未访问动作的覆盖。
- 控制检验：相同对手下比较无模型、无条件模型与条件模型。
- 递归检验：匹配环境步数和计算预算，分开深度与计算量。
- 稳健检验：保留错误层次、未知类型与变化的对手，同时报告偏离或交叉对战。

<a id="lesson-code"></a>

## 13 · 先验证响应核，再阅读神经工程

在完全给定的两动作任务上，比较条件响应与其边缘分布如何改变动作排序。没有拟合模型或运行 PR2 优化器。

```python
def conditional_response():
    """Isolate conditional vs marginal prediction, not the PR2 optimizer.

    rho(b|a) is a supplied hypothetical response model. It is not a claim
    that a simultaneous opponent observes a before choosing b. Fitting,
    variational inference and policy learning are deliberately not included.
    """
    q = [[4., 0.], [1., 2.]]
    rho = [[.1, .9], [.9, .1]]
    marginal = [.5, .5]  # rho averaged with a uniform prior over our action.
    return {"payoffs": q, "response": rho, "marginal": marginal,
            "marginal_values": [sum(x*y for x, y in zip(r, marginal)) for r in q],
            "conditional_values": [sum(x*y for x, y in zip(r, p))
                                   for r, p in zip(q, rho)]}
```

matrix_value、minimax_2x2 与 zero_sum_gap 检验矩阵评价；没有训练 PR2、GR2 或 NFSP 网络。

```python
def matrix_value(matrix, row_policy, column_policy):
    probabilities(row_policy); probabilities(column_policy)
    return sum(row_policy[i]*column_policy[j]*matrix[i][j]
               for i in range(len(row_policy)) for j in range(len(column_policy)))


def minimax_2x2(matrix):
    """Row maximizes, column minimizes. Optimize the lower envelope of two lines."""
    a, b = matrix[0]
    c, d = matrix[1]
    candidates = [0., 1.]
    denominator = a-c-b+d
    if denominator != 0:
        crossing = (d-c)/denominator
        if 0 <= crossing <= 1:
            candidates.append(crossing)
    lower = lambda p: min(p*a+(1-p)*c, p*b+(1-p)*d)
    p = max(candidates, key=lower)
    return [p, 1-p], lower(p)


def zero_sum_gap(matrix, row_policy, column_policy):
    row_best = max(dot(row, column_policy) for row in matrix)
    column_best = min(sum(row_policy[i]*matrix[i][j] for i in range(len(matrix)))
                      for j in range(len(matrix[0])))
    return row_best-column_best


def counterfactual_advantage(matrix, row_action, column_action, row_policy):
    probabilities(row_policy)
    baseline = sum(row_policy[i]*matrix[i][column_action] for i in range(len(row_policy)))
    return matrix[row_action][column_action]-baseline


def monotone_joint_greedy(local_values, weights):
    if len(local_values) != len(weights) or any(w < 0 for w in weights):
        raise ValueError('nonnegative mixing weights required')
    local_choice = tuple(max(range(len(q)), key=q.__getitem__) for q in local_values)
    joint = list(itertools.product(*(range(len(q)) for q in local_values)))
    value = lambda acts: sum(w*q[a] for w, q, a in zip(weights, local_values, acts))
    return local_choice, value(local_choice), max(map(value, joint))
```

可复制为独立 Python 文件运行；检查软值恒等式和有限差分，不是学习实验。

```python
from math import exp, log

def soft_response(values, alpha=1.0):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    peak = max(values)
    weights = [exp((q-peak)/alpha) for q in values]
    z = sum(weights)
    return [w/z for w in weights], peak + alpha*log(z)

q = [0.0, 1.0]
rho, free_value = soft_response(q)
expected = sum(p*v for p, v in zip(rho, q))
entropy = -sum(p*log(p) for p in rho if p > 0)
assert abs(free_value-expected-entropy) < 1e-12
# Differentiate F([a, 1-a]) at a = 0.3.
a, eps = 0.3, 1e-6
probs, _ = soft_response([a, 1-a])
finite = (soft_response([a+eps, 1-a-eps])[1]
          - soft_response([a-eps, 1-a+eps])[1]) / (2*eps)
assert abs(finite-(probs[0]-probs[1])) < 1e-8
print(rho, expected, entropy, free_value)
print("soft-value gradient:", finite)
```

| 原始入口 | 阅读位置 | 实现边界 |
| --- | --- | --- |
| ying-wen/gr2 | code/maci/get_agents.py；learners/mavb_ac.py | joint critic、SVGD 粒子、soft backup、replay 和 target；旧 TensorFlow 依赖需独立配置 |
| MultiLevelPolicy | policies/level_k_policy.py 的 actions_for | 交替自身/对手策略；部分对手分支 stop_gradient |
| GeneralizedMultiLevelPolicy | level_distribution 与 actions_for | 对 1…k 层确定动作加权；动作平均不同于先随机选一个策略 |
| OpenSpiel NFSP / CFR | python/pytorch/nfsp.py；python/algorithms/cfr.py | 平台实现，不标作原论文当年实验快照 |

复现应记录提交、依赖、实际输入权限、层数、粒子数、replay 预算与更新顺序。先在小型游戏验证数据路径，再做高维任务。能导入旧工程并不说明已经复现论文结果。

<a id="lesson-branches"></a>

## 14 · 从内部模型到持续适应与策略种群

对手模型可在局内依据历史推断，也可跨局积累参数或记忆。前者可能只是状态推断，后者涉及持久改变；应分别说明。固定网络增加内部推理深度，增加的是当次计算，不自动产生跨经历的持续学习。

长期面对新参与者，还需处理模型失配后的可塑性、身份未知时的状态构建、历史保留与资源预算。“对手总比自己少想一层”只是可检验假设。

策略种群解决哪些完整策略值得保留、怎样选择训练对手或伙伴、下一次应补哪个弱点。它可使用对手模型，却不要求每个成员递归推理。将内层响应放回评价、目标构建与改善的循环，见[开放式多智能体学习](multi-agent-populations.md)；联合探索与团队信用的困难则回到[合作多智能体学习](multi-agent.md)。

<a id="lesson-check"></a>

## 15 · 检查理解

- 问：自对弈必须共享网络吗？答：不需要。各角色的奖励和信息可以不同。
- 问：$\rho(a_{-i}\mid s,a_i)$ 证明对手看到本步动作吗？答：不能。真实权限由环境协议决定。
- 问：F 大于最大 Q 违反回报上界吗？答：不违反。F 包含正则项，不是原始奖励期望。
- 问：level-3 必然战胜 level-2 吗？答：不必然；模型失配、响应近似与资源成本都影响结果。
- 练习：将算例 α 改为 0.1 和 10。分别计算普通期望与软值，解释二者差异。
- 练习：两个策略各自连续两步总选左或总选右。比较每局抽一次策略和每步混合动作可产生的轨迹。



<a id="chapter-code"></a>

## 下载与运行

下载本页配套脚本后运行。精确条件评分、虚拟博弈与零和 gap 的机制检查，不是 PR2/GR2 神经训练。

[下载 marl_objectives_lab.py](../../examples/marl_objectives_lab.py)

```sh
python3 examples/marl_objectives_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Heinrich、Lanctot、Silver · Fictitious Self-Play（ICML 2015）](https://proceedings.mlr.press/v37/heinrich15.html)：阅读实现等价的行为平均与 XFP/FSP，注意到达概率权重。

- [Heinrich、Silver · Neural Fictitious Self-Play（2016）](https://arxiv.org/abs/1603.01121)：Algorithm 1 的按局模式抽样、两种记忆与平均策略评价。

- [Zinkevich 等 · Regret Minimization in Games with Incomplete Information（2007）](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)：反事实价值、局部到整体遗憾界与完美回忆条件。

- [Wen、Yang、Luo、Wang、Pan · PR2（ICLR 2019）](https://arxiv.org/abs/1901.09207)：条件响应的变分推断、Theorem 1/2 和 PR2-Q/PR2-AC；保留定理假设。

- [Wen、Yang、Wang · GR2（IJCAI 2020）](https://www.ijcai.org/proceedings/2020/58)：第 4 节层次模型与第 5 节实现近似；有限理性模型不是人类心理机制的实证证明。

- [Tian、Wen 等 · ROMMEO（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/85)：阅读经验对手先验、KL 正则与合作场景；它改变模型目标，不只是增加预测头。

- [ROMMEO 作者代码](https://github.com/rommeoijcai2019/rommeo)：论文指定的表格与 actor–critic 实现；不是任意对抗场景的稳健性保证。

- [Fu、Tian、Wen 等 · GSCU（ICML 2022）](https://proceedings.mlr.press/v162/fu22b.html)：阅读离线策略嵌入、线上后验与两策略选择；遗憾比较对象及信息权限须保留。

- [GSCU 作者代码](https://github.com/YeTianJHU/GSCU)：embedding_learning、conditional_RL 与 online_test 分别对应准备、响应训练和线上适应。

- [GR2 作者代码与理论附录](https://github.com/ying-wen/gr2)：原文指定工程，含 PR2 基线；附录 D 的低维动力学分析不是深度训练全局定理。

- [GR2 · 递归与层次混合实现](https://github.com/ying-wen/gr2/blob/master/code/maci/policies/level_k_policy.py)：检查 stop_gradient 与确定动作加权，勿称精确随机层次混合。

- [GR2 · 条件响应与 actor–critic](https://github.com/ying-wen/gr2/blob/master/code/maci/learners/mavb_ac.py)：检查 critic、SVGD、粒子软聚合与 replay 字段；仍需独立运行验证兼容性。

- [OpenSpiel · NFSP](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/nfsp.py)：公开平台实现；阅读按局策略抽样、reservoir 和监督损失。

- [OpenSpiel · CFR](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py)：对照 counterfactual reach、regret matching 和 average_policy。

- [Lanctot 等 · Policy-Space Response Oracles（2017）](https://arxiv.org/abs/1711.00832)：将训练对手与新增响应组织为种群循环，和内部递归是不同维度。


<a id="marl-mechanism-experiments"></a>

## 小型实验：把更新规则与评价对象分开

全部结果来自完全列举的有限博弈，没有采样误差或置信区间。它们检验具体机制，不是大型神经算法复现。

### 条件响应与边缘化

价值表((4,0),(1,2))。对手边缘分布(1/2,1/2)给出我方动作值2与1.5；给定条件响应模型两行为(0.1,0.9)与(0.9,0.1)，动作值变成0.4与1.1，排序反转。这不证明响应模型正确或有因果解释。PR2还需要条件模型、critic与策略训练；GR2还需要明确递归深度和底层行为。

### 运行与复核

```bash
python3 examples/marl_objectives_lab.py test
python3 examples/marl_objectives_lab.py demo --out results/marl-objectives
```

[源码](../../examples/marl_objectives_lab.py) · [全部数值与源码哈希](https://yingwen.io/crl-code/marl-objectives/results.json)

仅依赖Python标准库，含15项机制检查。图不用于排序大型算法。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](../tabular/dynamic-programming.md) → [策略梯度定理](../approximation/policy-gradient.md) → [TRPO 与 PPO 的近似](trust-region.md) → [持续控制的比较器](../../textbook/control.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [智能体状态与递归学习](../../textbook/state.md)
- [元学习与学习规则的适应](../../textbook/meta.md)
- [实验设计、统计与算法测试](../../textbook/experiments.md)
- [持续学习的智能体架构](../../textbook/architectures.md)

对应原始材料：Fictitious Play 与 Fictitious Self-Play；NFSP 与 CFR；PR2（ICLR 2019）；GR2（IJCAI 2020）。本文为原创讲解，原书、论文与上游代码保留各自许可。
