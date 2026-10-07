# 开放式多智能体学习：评估、目标构建与策略改善

怎样由交互评估构建每轮学习目标，使竞争或合作能力沿明确的评价准则持续改善？

## 本章内容

- 沿竞争自对弈、策略种群与 PSRO 理解开放式多智能体学习。
- 建立评估、目标构建、响应学习与重新评估的循环，并定义所追求的单调提升。
- 理解 COLE 与 HOLA 怎样把这条思路扩展到合作伙伴与团队组合。
- 区分固定对手回报、可利用度、种群安全价值与陌生伙伴泛化。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 联合策略与局部信息

各参与者按自己的观察历史行动。集中训练不等于执行时可获得所有信息。

### 最佳响应

固定其他参与者后，使自身期望回报最大的策略。实际 RL 通常只能给出近似响应。

### Nash 均衡

没有参与者能通过单方偏离提高自身收益。合作中的均衡不必是最高团队回报。

<a id="problem-definition"></a>

## 本章的问题定义

开放式多智能体学习：多个参与者反复交互，学习系统依据已有策略的交互评估，不断产生下一轮对手、伙伴和响应目标。先沿竞争自对弈与种群学习建立框架，再研究合作场景中的兼容性与团队组合。主体采用固定奖励与动力学、可重置回合；生命期与有限资源问题另作扩展。

### 给定条件与符号

- 各角色允许使用的观察、历史与通信。
- 任务奖励、回合长度、初始分布与交互预算。
- 初始策略池，以及明确的外部评价准则。

### 需要求解的对象

构建能暴露当前能力缺口的学习目标，训练并保留有用响应，选择可提交的策略或组合；在固定评价定义下建立或检验逐轮改善。

### 信息与数据权限

外层评估与课程模块可读取训练策略之间的经验回报。执行策略不得因此获取对手私有状态、未来动作或测试伙伴身份。

$$
J_k(\pi)=\mathbb E_{\xi\sim\mu_k}[u(\pi,\xi)]+\lambda_k D_k(\pi),\qquad \mathcal E(\mathcal O_{k+1})\ge\mathcal E(\mathcal O_k)
$$

左式是可随轮次改变的训练目标，μ 是对手或伙伴分布，D 是可选的多样性项。右式是希望建立的单调改善性质，不是假定已成立的算法定理。O 是明确声明的输出，E 是各轮相同、越大越好的评价准则。

### 成立条件与解的含义

- 仅在两人零和分支使用极小极大性质。
- 精确响应、完整对手空间与无遗憾条件逐项核对。
- 课程生成和最终评估分开；双方消耗的资源均计入预算。

判断准则：说明改善对象和评价空间；给出相应条件下的保证，或用独立数据检验逐轮改善及退步。同步报告估计误差、测试覆盖和全部交互成本。

### 适用边界

- 不把种群增长直接视为有限资源单一生命的完整解。
- 不把零样本合作直接称为部署阶段的持续参数学习。
- 不将不同单调性结果合并成“每一代都更强”。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [探索与经验选择](../../textbook/exploration.md)：不只决定当前动作，也决定与谁交互以发现学习机会。

- 改变评价目标 · [持续控制与学习智能体比较](../../textbook/control.md)：参与者分布改变后，期望回报目标随之改变。

- 组合不同学习问题 · [知识保留与再适应](../../textbook/retention.md)：策略档案保存旧响应，删除或蒸馏会改变可用组合。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

参与者分布决定当前学习目标；策略更新又会改变下一轮可选的参与者。没有分清评价空间，就可能把池内成功误当作稳健性或泛化。

### 本章的核心思路

用评估发现能力缺口，把缺口转成下一轮学习目标，再训练响应并重新评估。目标构建连接当前不足与下一步学习；保留旧解和选择新输出为建立单调改善提供可能。

1. [定义进步并评估现状](multi-agent-populations.md#lesson-evaluation)：声明提交策略还是种群，固定 Gap、安全价值或测试伙伴回报的定义。

2. [把竞争弱点变成响应目标](multi-agent-populations.md#lesson-psro)：PSRO 用元策略构建训练对手；响应学习后扩池并重新评估。

3. [学习课程生成规则](multi-agent-populations.md#lesson-curricula)：NAC 将多轮训练后的评价反传给课程生成器，区分内外层时钟。

4. [将兼容性缺口转为伙伴分布](multi-agent-populations.md#lesson-cole)：COLE 的求解器评价策略间关系，训练器优化给定伙伴课程。

5. [评价完整伙伴组合](multi-agent-populations.md#lesson-hola)：HOLA 用高阶组合表示两两关系无法表达的合作机会。

结论与条件：嵌套己方集合的最佳安全价值不下降；精确有限零和 DO 有相应终止证书。平均无遗憾界、局部偏好结果与实际神经训练分别有条件，均不自动给出全寿命或陌生伙伴回报单调改善。

### 相关方法改变了什么

- 冻结伙伴／对手分布：评价简单，但不能主动寻找新的弱点或兼容性缺口。

- 无差别增加种群规模：增加资源，不一定增加响应覆盖，也未定义哪些策略值得保留。

- 仅评价最后网络：忽略种群的组合能力，也忽略整个学习期间的成本。


<a id="lesson-setting"></a>

## 1 · 开放式多智能体学习：评估怎样产生下一轮目标

给定对手与奖励，强化学习回答怎样改进策略。对手也在学习时，还要回答：下一轮应当回应谁，针对什么弱点学习，怎样判断整体能力提高了？开放式多智能体学习把这些问题纳入学习系统，以交互评估持续产生新的学习目标。

这一方向的一条主流脉络来自竞争场景：自对弈用当前对手产生挑战；历史策略和种群保留已遇到的行为；PSRO 进一步把策略间评估、对手混合与响应训练组织成循环。竞争暴露的克制关系由此成为后续学习的材料。[PSRO 原文](https://arxiv.org/abs/1711.00832)

合作也需要构建学习目标，但缺口从“被谁克制”变为“与谁不能配合”。温颖及合作者的 COLE 将这条思路扩展到合作兼容性：评估策略间关系，生成伙伴课程，再学习新的合作策略。HOLA 进一步处理多个伙伴的组合关系。[COLE](https://proceedings.mlr.press/v202/li23au.html) · [HOLA](https://arxiv.org/abs/2409.08767)

$$
u_i(\pi_i,\pi_{-i})=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^tR^i_{t+1}\right]
$$

$u_i$ 是固定环境、起点、回合长度和各方策略后的期望回报。策略可以依赖局部历史，不要求物理状态完全可见。

$$
\widehat U_k=\operatorname{Evaluate}(\mathcal P_k),\quad \mu_k=\operatorname{Target}(\widehat U_k,\mathcal P_k),\quad \pi_{k+1}=\operatorname{Learn}(J_k),\quad \mathcal P_{k+1}=\operatorname{Retain}(\mathcal P_k,\pi_{k+1})
$$

P 是策略档案；U 是交互评估；Target 将弱点或兼容性缺口转成训练分布。Learn 使用有限预算求响应，Retain 保留所需策略；随后重新评估。评估器、目标构建器和学习器具有不同职责。

例如，已有策略能战胜对手甲，却持续输给乙。评估先定位乙揭露的弱点；目标构建提高相关挑战的训练权重；响应学习形成新策略；重新评估检查它是否弥补弱点，也检查旧能力是否保留。合作中可以把乙换成难以配合的伙伴。关键不是盲目扩大种群，而是让每轮学习针对可识别的能力缺口。

| 问题 | 学习对象 | 最终检验 |
| --- | --- | --- |
| 固定团队合作 | 共同训练的参与者 | 遵守执行信息限制时的团队回报。 |
| 竞争种群学习 | 新响应与历史对手混合 | 对完整或明确测试对手空间的可利用度。 |
| 开放合作 | 学习者与训练伙伴组合 | 未共同训练伙伴上的收益与失败分布。 |
| 部署期持续学习 | 同一学习者长期遇到新旧参与者 | 全寿命收益、适应损失、记忆与恢复成本。 |

这里专指 open-ended multi-agent learning。开放的是由交互结果不断构建后续学习问题的过程，不要求奖励函数或物理世界不断改变。它是开放式人工智能的一条具体研究线，不是整个 open-ended AI 的定义。它也不等同于 CRL：种群可在可重置回合中训练新策略；CRL 还需声明同一学习者的生命期、资源与知识更新。

<a id="lesson-evaluation"></a>

## 2 · 评价策略，与评价一个种群，不是同一问题

单调提升是本章希望建立的性质。先把第 $k$ 轮提交的对象记为 $\mathcal O_k$，并固定越大越好的评价函数 $\mathcal E$。对象可以是一个策略、一个混合，或一个允许重新选择混合的种群。逐轮改善要求 $\mathcal E(\mathcal O_{k+1})\ge\mathcal E(\mathcal O_k)$；它比仅要求最终一轮更好更强。

| 学习问题 | 可选择的评价准则 | 怎样理解进步 |
| --- | --- | --- |
| 竞争中的提交策略对 | 完整游戏的负 Gap | 更少的单方偏离获利空间。 |
| 竞争中的策略档案 | 面对固定完整对手空间的最佳安全价值 | 可从档案组合出的最强防守不退步。 |
| 开放合作中的提交策略 | 固定伙伴分布上的期望团队回报 | 在相同配合要求下，合作收益提高。 |

训练分布可以改变，比较进步的准则仍需可比。若同时更换考题和策略，分数变化混合了两个原因。保留旧策略可使旧解仍然可选；是否能找到并提交更好的解，则取决于评估和优化精度。以下先给出确实成立的集合层面结论，再说明它与最新策略改善的区别。

先取两人零和游戏，行方最大化 $u$，列方最小化 $u$。$\Pi_1,\Pi_2$ 是完整策略集合；$\mathcal P_1,\mathcal P_2$ 是已发现的有限池；$x,y$ 是完整策略上的混合。每回合抽一个策略，与每一步重抽策略，不是同一个执行过程。

$$
\begin{aligned}\operatorname{Gap}(x,y)&=\max_{p\in\Delta(\Pi_1)}u(p,y)-\min_{q\in\Delta(\Pi_2)}u(x,q),\\v(\mathcal P_1)&=\max_{x\in\Delta(\mathcal P_1)}\min_{q\in\Delta(\Pi_2)}u(x,q).\end{aligned}
$$

Gap 衡量提交策略对的单方获利空间；部分文献将其一半称为 exploitability。种群安全价值则允许重新选择池内混合，但仍面对完整对手空间。

$$
\mathcal P_1\subseteq\mathcal P'_1\quad\Longrightarrow\quad v(\mathcal P'_1)\ge v(\mathcal P_1)
$$

旧混合仍可行，因此扩张己方集合不能降低最佳安全价值。这是集合包含的结论，不是神经网络训练定理。

它没有保证实际求解器找到这个最优混合，也没有保证最新单策略更强。若同时扩大对手池，受限游戏的数值可能下降，因为新对手暴露了旧漏洞。只在自己的档案中评价，会把尚未发现的反例当作不存在。

普通剪刀石头布中，双方池里只有石头时，池内 Gap 为零；完整游戏中双方都出石头的 Gap 为二。另一个加权游戏甚至允许“加入精确响应后，受限均衡的完整 Gap 变大”，下一节给出手算。

<a id="lesson-psro"></a>

## 3 · DO 与 PSRO：评价、生成反例、扩张策略集合

竞争中的目标构建有一个直接依据：当前策略仍能被什么行为利用？用这样的对手组织响应学习，再把学到的行为纳入种群，可以逐步补齐战略覆盖。DO 与 PSRO 将这个直觉变成可执行的评估—响应循环。

Double Oracle 从受限集合开始。先解受限游戏，再到完整策略空间寻找有利偏离。PSRO 将完整行为策略作为元博弈的动作，用 RL 近似求响应。原框架允许不同元求解器；这里只推导两人零和的 Nash 版本。[PSRO 原文](https://arxiv.org/abs/1711.00832)

$$
\begin{gathered}A^k_{ab}=u(\pi_1^a,\pi_2^b),\qquad(x_k,y_k)\in\operatorname{NE}(A^k),\\\pi_1^{k+1}\approx\arg\max_{\pi_1}u(\pi_1,y_k),\qquad\pi_2^{k+1}\approx\arg\min_{\pi_2}u(x_k,\pi_2).\end{gathered}
$$

矩阵通常由多次交互估计。固定对方元策略并从中抽完整对手，响应训练就形成一个可用单智能体 RL 求解的子问题。

**算法：响应训练与响应评价使用不同随机数据，避免把训练噪声当成新的弱点。**

1. 保留双方策略池，以及每个策略版本的身份。
1. 评估新增策略对，记录 payoff、样本数与估计误差。
1. 在受限游戏中求元策略 x、y。
1. 固定 y 训练行方响应；固定 x 训练列方响应。
1. 独立评价新响应的获利幅度，再加入档案。
1. 重新求元策略，并在完整或独立对手集合上评价。

$$
A=\begin{pmatrix}0&-1&1\\1&0&-10\\-1&10&0\end{pmatrix}
$$

行方最大化。初始双方池均为 {0}，唯一池内均衡是动作零，完整 Gap 为 1−(−1)=2。动作一是对动作零的精确最佳响应。

双方加入动作一后，受限矩阵为 $\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$，唯一均衡双方都选一。面对完整游戏，它的 Gap 为 $10-(-10)=20$。没有采样误差，也没有神经优化失败；变化来自受限均衡没有防守尚未纳入的动作二。

有限两人零和游戏中，若受限均衡与完整最佳响应都精确，双方均无有利偏离就是终止证书。近似 RL 没找到新策略，只说明本次搜索失败；不能证明不存在更好的响应。

$$
\hat B-\hat C\le\operatorname{Gap}(x,y)\le\hat B-\hat C+\epsilon_1+\epsilon_2
$$

令 $\hat B=u(\hat p,y)$、$\hat C=u(x,\hat q)$ 是找到的响应的真实期望收益。若已知行方响应距完整最大值至多 $\epsilon_1$，列方响应距完整最小值至多 $\epsilon_2$，便得到上下界。未知 oracle 误差时只有左边，找不到获利偏离不构成上界证书；若 $\hat B,\hat C$ 也来自有限评价，还需计入其误差。

<a id="lesson-epsro"></a>

## 4 · EPSRO：响应与对手混合共同更新

标准 PSRO 分开进行元游戏评估和冻结混合上的响应训练。EPSRO 使用 unrestricted–restricted game，简称 URR：一方搜索完整策略类，另一方只在旧池中混合，两者共同更新。URR 是问题设定，不是另一篇独立算法论文。[EPSRO](https://arxiv.org/abs/2202.00633)

$$
\max_{p\in\Delta(\Pi_1)}\min_{y\in\Delta(\mathcal P_2^k)}u(p,y)
$$

行方不只回应冻结权重；列方也调整旧策略混合，继续暴露行方弱点。反向的 URR 可用于另一方。

实践中，一边用 RL 改进 response，一边以在线无遗憾方法更新元策略；旧信息可用于 warm start，响应可以流水并行。“减少完整元博弈评估”不等于无需交互：训练和收益估计仍消耗数据。

$$
\operatorname{Gap}_{\rm URR}(\bar p_T,\bar y_T)\le\frac{R_1(T)+R_2(T)}{T}
$$

有限双线性零和 URR 中，Gap 的行方最大化遍历完整集合，列方最小化仅遍历该轮受限池。若双方在这些集合上的累积外部遗憾为 R，时间平均策略满足此界。它不是对双方完整策略空间的无条件 Gap 界。

$$
\begin{aligned}R_1&=\max_p\sum_{t=1}^T u(p,y_t)-\sum_{t=1}^T u(p_t,y_t),\\R_2&=\sum_{t=1}^T u(p_t,y_t)-\min_y\sum_{t=1}^T u(p_t,y),\\\frac{R_1+R_2}{T}&=\max_pu(p,\bar y_T)-\min_yu(\bar p_T,y).\end{aligned}
$$

先在同一轮固定的双方可行集合中定义累积外部遗憾；相加时实际对局收益相消，双线性把时间平均移入 u。精确遗憾给出等式，遗憾上界给出上面的不等式。扩张集合、近似收益或非线性参数平均不能不经分析直接代入。

平均策略界不保证最后一次参数更新。EPSRO 的精确 URR 均衡、嵌套集合及遗憾分析也不能替换成任意深度 RL oracle。声称单调改善时，必须说明评价对象、策略空间、求解精度，以及输出是平均策略还是最新网络。

<a id="lesson-diversity"></a>

## 5 · 多样性：去过哪里，与能应对谁

两个策略可以走不同路线，却输给同一类对手；也可以只有一个关键动作不同，却克制不同对手。参数距离、占据分布和响应向量不是同一种多样性。Diverse-PSRO 研究 payoff 表征上的 DPP 指标；BD/RD 区分行为与响应两个层面。[Diverse-PSRO](https://proceedings.mlr.press/v139/perez-nieves21a.html) · [BD/RD](https://arxiv.org/abs/2106.04958)

$$
a(\pi)_j=u(\pi,\pi_2^j),\qquad D_R(\pi)=\min_{w\in\Delta(\mathcal P_1^k)}\|a(\pi)-(A^k)^\top w\|_2^2
$$

响应多样性衡量新 payoff 向量到旧向量凸包的距离。它依赖作为坐标的参考对手，不能直接代表所有未知对手。

$$
J_k(\pi)=u(\pi,y_k)+\lambda_B D_f(\rho_{\pi,y_k}\Vert\rho_{x_k,y_k})+\lambda_R D_R(\pi)
$$

这里用共同教学记号，ρ 是归一化折扣占据分布。若原文采用占据测度下平均奖励，权重需按回报尺度调整。加正则后不再是纯任务收益最佳响应。

凸包外也可能有很差的策略：新向量每个坐标都更低，仍可能离凸包很远。多样性要服务于发现有效响应，而不能自己充当最终成绩。复制策略能增加人数，却不增加可表达的混合。

合作的结构化探索与此相关，但尺度不同。Q-DPP 在联合行动层组织探索；BD/RD 在完整策略种群层组织探索。前者不自动生成稳健课程，后者也不自动解决轨迹内的团队信用分配。[Q-DPP](https://proceedings.mlr.press/v119/yang20i.html)

<a id="lesson-curricula"></a>

## 6 · NAC：直接学习怎样安排下一轮课程

如果响应训练只有有限预算，最适合学习的对手混合未必是当前受限 Nash。NAC 参数化元求解器，再根据若干轮后的可利用度训练它。[Neural Auto-Curricula](https://arxiv.org/abs/2106.02745)

$$
\mu_k=f_\psi(A^k),\quad\theta_{k+1}=\operatorname{BRLearn}(\theta_k,\mu_k),\quad\min_\psi\mathbb E_{G\sim\mathcal D}\operatorname{Gap}_G(x_T(\psi),y_T(\psi))
$$

元参数决定课程；课程影响响应；响应又改变未来矩阵。外层评价必须考虑这条依赖，不是只对当前对手概率做一次动作层策略梯度。

原文研究展开优化、元梯度和进化策略等途径。内层在一个游戏中增加策略，外层跨训练游戏更新课程生成器。跨游戏泛化与一个游戏中的逐轮改善不同。

固定外部评价仍然存在。自动学习的是为它服务的课程规则，不是凭空创造最终好坏标准。展开长度、训练游戏分布和响应质量限定了结论。[作者代码](https://github.com/waterhorse1/NAC)

<a id="lesson-composition"></a>

## 7 · 策略生成：继承、融合与信息状态级组合

| 方法 | 改变哪个环节 | 必须保留的边界 |
| --- | --- | --- |
| Fusion-PSRO | 按元策略权重融合历史参数，再训练 response | 参数平均不等于策略概率混合；网络的局部对齐条件重要。 |
| Conflux-PSRO | 状态级路由调用不同子策略，再蒸馏 | 路由和子策略同时更新，不是固定 MDP 的表格算法。 |
| XDO / NXDO | 在信息状态级组织受限游戏与组合 | 相邻团队工作；神经近似不自动继承表格 XDO 的界。 |

$$
\theta_{\rm init}=\sum_jx_k(j)\theta_j,\qquad\pi_{\theta_{\rm init}}\ne\sum_jx_k(j)\pi_{\theta_j}\quad\text{一般情况下}
$$

Fusion-PSRO 改的是 response 的起点。即使权重来自均衡，也不能据此宣称融合网络仍是均衡或拥有所需状态覆盖。

$$
\pi_{\rm route}(a\mid h)=\sum_j\mu(j\mid h)\pi_j(a\mid h)
$$

逐信息历史路由与回合开始一次抽取不同，后者保留跨时间相关性。若路由读取部署者不允许获得的全局状态，则改变了原问题。

XDO/NXDO 由 McAleer 等提出，不属于温颖共同作者系列。它在信息状态上组织组合，与 EPSRO 联动元策略和 response 的做法不同。组合扩大候选类，但近似优化仍可能失败；应给路由、蒸馏和评估同等预算。[Fusion-PSRO](https://doi.org/10.3233/FAIA251106) · [Conflux-PSRO](https://arxiv.org/abs/2410.22776) · [XDO/NXDO](https://arxiv.org/abs/2103.06426)

<a id="lesson-cole"></a>

## 8 · COLE：把合作不兼容转成下一轮伙伴课程

前面的竞争主线通过评估寻找克制关系，再据此构建训练目标。转向合作后，核心循环保留，评价关系改变：需要发现哪些约定或伙伴组合尚不能兼容，并使后续策略学会配合。这是 COLE 扩展开放式多智能体学习的出发点。

共同回报矩阵 $C=\left(\begin{smallmatrix}3&0\\0&2\end{smallmatrix}\right)$ 有两种成功约定。总选左和总选右在各自自博弈中表现良好，彼此组队却得零。提高 self-play 分数不等于解决 cross-play。

COLE 用图记录策略间合作收益。节点是策略，边权是互动评价；不是物理连接。求解器识别兼容性缺口，生成伙伴分布，再由 RL 训练新策略。新的互动结果重新进入图，使评价、目标构建与策略改善连接起来。原始论文是 ICML 2023，JAIR 2024 为扩展。[COLE](https://proceedings.mlr.press/v202/li23au.html)

$$
M^k_{ij}=u(\pi_i,\pi_j),\quad q_k=\operatorname{Solver}(M^k),\quad J_k(\pi)=\mathbb E_{j\sim q_k}u(\pi,\pi_j)+\alpha u(\pi,\pi)
$$

第一项学习与难配合伙伴合作，第二项保留自我配合目标。外部任务奖励可以不变，训练伙伴分布持续改变。

COLE-SV 的 graphic Shapley value 评价策略在合作图中的作用，以构建课程；不是将每步团队奖励分摊给动作。COMA baseline、团队成员 Shapley 信用与种群课程评价是三个不同对象。

局部偏好收敛分析要求 response oracle 满足规定质量，不保证任意 PPO 调用都产生合格响应，更不保证所有陌生人类回报单调提高。当前池内兼容性、冻结测试伙伴表现和人机评价应分开。[JAIR 扩展](https://doi.org/10.1613/jair.1.15884)

还需审查原证明的一个具体推步。ICML 版本 Appendix B 写出偏好中心性 $\eta_T=\eta_0\prod_{t=0}^{T-1}(1-\alpha_t)$，其中 $\alpha_t$ 是当轮相对改善量，不是 self-play 的混合超参。从这个乘积推到零，这一步还需要累计改善条件，例如 $\sum_t\alpha_t=\infty$（或统一正下界）。仅有每轮 $0<\alpha_t<1$ 不够：取 $\alpha_t=1/(t+2)^2$，乘积等于 $(T+2)/(2(T+1))$，极限为一半。正文 Theorem 4.4 未给出这样的统一下界；每代有限、随后扩张的策略池也不自动提供它。因此这里保留 COLE 构建课程的机制与 Overcooked 实验观察，不把该推步或 PPO 近似视为无条件收敛证书。[原文 §4.1–4.2、Appendix B](https://proceedings.mlr.press/v202/li23au/li23au.pdf)

<a id="lesson-hola"></a>

## 9 · HOLA：伙伴组合具有两两关系之外的结构

COLE 说明怎样从两方合作关系构建课程。多个伙伴同时参与时，目标构建还需知道“哪些组合”值得学习；不能只把所有两两评价分别提高。HOLA 将评估单元扩展为整个团队组合，再用评估结果形成下一轮课程。

一个任务同时需要侦察者、搬运者和协调者。任意两两组合都可能失败，三者一起才能成功。此时“某个伙伴好不好”不能独立回答；学习机会属于整个组合。

$$
W_k(i_1,\ldots,i_m)=u(\pi_{i_1},\ldots,\pi_{i_m}),\qquad J_k(\pi)=\mathbb E_{\boldsymbol z\sim q_k}[u(\pi,\boldsymbol z)]
$$

超边对应一个策略组合，权重为共同回报。课程分布选择伙伴元组，而非必然独立抽样的单个伙伴。

HOLA-Drone 用超图及偏好关系组织评估，生成下一轮伙伴课程，研究多无人机与未见伙伴的配合。超图是博弈关系的表示，不等于策略必须使用超图神经网络。[HOLA-Drone](https://arxiv.org/abs/2409.08767)

组合数随种群和团队规模增长。未评价的超边不能当作零收益，共享成员的组合也不是独立样本。评价采样和不确定性应计入算法成本。

后续 Multi-Robot Open Adaptive Teaming 考察环境、伙伴及团队规模变化。无微调迁移表明已训练策略可以适应；它不等于同一实体在部署中持续更新参数。[后续原文](https://arxiv.org/abs/2607.04972)

<a id="lesson-test-distribution"></a>

## 10 · 独立评价：训练课程不能同时充当考卷

训练分布由方法选择，测试分布应由研究问题确定。若每种方法挑自己的容易伙伴，平均回报不能比较。即便共用伙伴，不同伙伴能实现的最佳团队收益也可能不同。

$$
\operatorname{Regret}(\pi,z)=\max_{\pi'}u(\pi',z)-u(\pi,z)
$$

伙伴条件 regret 比较同一个伙伴下的收益差。复杂环境的最大值通常未知，只能用独立训练的近似响应作参考。

ZSC-Eval 通过行为偏好奖励生成候选伙伴，按所需最佳响应的差异筛选集合，再用 BR-Prox 等衡量适应差距。它使“哪些配合能力还没有学会”成为可研究的评价问题。[ZSC-Eval](https://arxiv.org/abs/2310.05208)

参考响应并不等于真实最优。BR-Prox 分母近零、收益可正可负或参考训练不足时，不能机械套用比值。必须同时报告原始回报、分伙伴结果、区间和参考训练预算。测试生成过程也只是部署分布的代理。

人类会主动迁就 AI。首次与重复配合、角色和沟通权限、主观负担都应记录，以区分 AI 适应、人类适应和共同适应。短期人机高分不直接证明长期协作稳定。

<a id="lesson-series"></a>

## 11 · 扩展同一条主线：数据限制、团队结构与一般和目标

开放式多智能体学习不只面临“再增加一个策略”的问题。无法继续采样时，响应必须受已有数据约束；双方各有一支团队时，需要声明能否相关地选择联合策略；利益不完全对立时，则要改变解概念。以下作品分别扩展评估、目标构建或响应学习的条件。

| 问题 | 温颖及合作者的工作 | 核心思想与边界 |
| --- | --- | --- |
| 无法继续与对手交互 | [Offline Fictitious Self-Play](https://arxiv.org/abs/2403.00841) | 重加权固定数据来近似不同对手下的经验，再用离线 RL 学响应；重要性采样不能创造数据未覆盖的动作。 |
| 对手能整队协调偏离 | [Leveraging Team Correlation](https://arxiv.org/abs/2403.00255) | 区分个人偏离与团队相关偏离，以受限团队解和顺序相关机制组织搜索；安全性只相对于声明的偏离空间。 |
| 队友角色不同 | [Heterogeneous-PSRO](https://arxiv.org/abs/2410.01575) | 扩展异质团队策略的表达与顺序响应；表示限制、理想改进界与有限训练误差需要分别分析。 |
| 既非共同奖励也非零和 | [Stochastic Games with Potentials](https://proceedings.mlr.press/v139/mguni21a.html) | 利用特殊势结构关联个人改进和共同势函数；不是任意一般和游戏都能化为单目标优化。 |
| 采样、训练、评估彼此嵌套 | [MALib](https://jmlr.org/papers/v24/22-0169.html) | 以任务调度和 Actor–Evaluator–Learner 支持种群课程；并行吞吐是系统指标，不是学习质量保证。 |

$$
\sigma_{\rm team}\in\Delta\!\left(\prod_i\Pi_i\right)\quad\text{与}\quad(\sigma_i)_i\in\prod_i\Delta(\Pi_i)
$$

左侧允许对联合策略进行相关随机选择；右侧仅独立随机化。两个集合对应不同的协作与对手能力假设。事前协商不等于允许执行时共享私有观察。

若对手被允许整队协调改变策略，只测试单个成员偏离不够。反之，不能把己方有特权通信、对方只能独立随机化时的优势称为公平的均衡比较。异质性也不等于必须完全独立参数：共享网络若有充分角色条件，可能表达不同角色；应检验具体函数类。

Off-FSP 属于数据受限的竞争学习，不能因为使用历史经验就称为在线 CRL。势博弈是结构性分支，不能将其收敛结论外推给所有开放合作。上述研究分别改变数据协议、策略可行集合、评价或计算组织，因而需要不同实验。

<a id="lesson-branches"></a>

## 12 · 开放式多智能体学习与 CRL：评价整个学习过程

开放式多智能体学习研究怎样由交互生成新的学习目标，并沿明确准则改善能力。把它放入 CRL 的生命期设定后，还要把形成能力的过程本身纳入评价。最终档案更强与同一智能体一生获得更高收益，是相关但不同的目标。

| 现有机制 | 新增困难 | 所需对照 |
| --- | --- | --- |
| 策略池不断增加 | 参数、检索和评估开销不能无限增长 | 固定档案容量，比保留、删除、蒸馏后的旧弱点。 |
| 每回合抽取伙伴 | 伙伴会返回、漂移或在途中离开 | 控制变化可观测性，测恢复时间与历史利用。 |
| 冻结策略零样本适应 | 状态、记忆和参数是不同更新载体 | 分别冻结各载体，不把活动变化当作无适应。 |
| 只测最终 checkpoint | 探索、更新和错误在生命中有成本 | 全寿命回报、适应损失、失败与恢复代价。 |

合作结构化探索决定一次经历发现什么；种群课程决定下一轮遇到什么；持续学习还要决定保留哪些知识、何时更新和付出多少代价。三层相连，但一层的成功不能代替其他层的证据。

可以先固定物理规则，只让伙伴约定变化；再固定伙伴，改变动力学。最后组合二者，研究有限记忆下的状态、预测和控制学习，避免把所有困难笼统归为非平稳。

<a id="lesson-code"></a>

## 13 · 可运行的评价反例，与原始工程入口

精确小博弈：fictitious play 的平均混合与最新响应、完整 gap、伙伴分布变化，以及精确最佳响应扩池后的反例。不是神经 PSRO 复现。

```python
def saddle_gap(matrix, row, col):
    """Full-game zero-sum gap, not win rate against one training opponent."""
    distribution(row)
    distribution(col)
    upper = max(sum(a*b for a, b in zip(r, col)) for r in matrix)
    lower = min(sum(row[i]*matrix[i][j] for i in range(len(row)))
                for j in range(len(col)))
    return upper-lower


def fictitious_play(rounds=300):
    """Symmetric RPS: BR to opponent's empirical average, not latest action.

    Start with one rock observation. Both roles have the same population
    by game symmetry; a best response is computed by full enumeration.
    Tie-breaking selects the lowest action index. Report BOTH the latest
    pure policy and the empirical mixture, which are different objects.
    This is exact matrix-game fictitious play, not NFSP or neural PSRO.
    """
    counts = [1, 0, 0]
    rows = []
    for k in range(rounds+1):
        mixture = [c/sum(counts) for c in counts]
        scores = [sum(a*b for a, b in zip(r, mixture)) for r in RPS]
        action = max(range(3), key=lambda i: scores[i])
        pure = [float(i == action) for i in range(3)]
        rows.append({"round": k, "mean_gap": saddle_gap(RPS, mixture, mixture),
                     "latest_gap": saddle_gap(RPS, pure, pure),
                     "mixture": mixture, "best_response": action})
        counts[action] += 1
    return rows


def partner_shift():
    """Coordination reward I[a=b]. Fixed and changing objectives differ."""
    matrix = [[1., 0.], [0., 1.]]
    policies = {"train_specialist": [1., 0.], "balanced": [.5, .5]}
    partners = {"training": [.9, .1], "held_out": [0., 1.]}
    return {name: {group: value(matrix, pi, mu) for group, mu in partners.items()}
            for name, pi in policies.items()}


def expanding_pool_counterexample():
    """Exact double-oracle expansion can increase full-game exploitability.

    Both pools initially contain action 0 only. Action 1 is each player's
    exact best response. The expanded 2x2 restricted equilibrium is (1,1).
    But the absent action 2 exploits it with magnitude 10. Pool inclusion
    alone does not make the CURRENT restricted equilibrium's gap monotonic.
    """
    matrix = [[0., -1., 1.], [1., 0., -10.], [-1., 10., 0.]]
    return {"matrix": matrix, "initial_gap": saddle_gap(matrix, [1., 0., 0.], [1., 0., 0.]),
            "expanded_gap": saddle_gap(matrix, [0., 1., 0.], [0., 1., 0.])}
```

标准库运行。枚举期望和精确最佳响应，不含采样置信区间；图用于检验机制而非论文性能。

```bash
python3 examples/marl_objectives_lab.py test
python3 examples/marl_objectives_lab.py demo --out results/marl-objectives
```

| 原始入口 | 重点代码接口 | 边界 |
| --- | --- | --- |
| [OpenSpiel PSRO](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/psro_v2) | 元游戏、求解器、response oracle、evaluation | 通用 PSRO 不是 EPSRO 作者实现。 |
| [NAC](https://github.com/waterhorse1/NAC) | 课程网络、展开内循环、外层评价 | 元梯度不提供最后策略的普遍单调保证。 |
| [NXDO](https://github.com/indylab/nxdo) | 信息状态 meta-action 与受限求解 | 神经版本不自动满足表格定理。 |
| [COLE-Platform](https://github.com/liyang619/COLE-Platform) | cole_training 与 baseline_training 分支 | main 分支不能直接当作 COLE-SV 训练器。 |
| [ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval) | 伙伴生成、筛选、参考响应、分伙伴评价 | 有限伙伴池不代表所有部署人群。 |
| [MALib](https://github.com/sjtu-marl/malib) | 种群与经验管理基础设施 | 清单中未勾选的算法不是实现证据。 |

EPSRO、BD/RD、Fusion-PSRO、Conflux-PSRO 与 HOLA 的原文见参考。这里不以背景仓库替代尚未确认的完整作者实现，也不把小矩阵核的正确性当作大规模训练性能证据。

<a id="lesson-check"></a>

## 14 · 练习：同一结果为什么会有不同解读

- 问：扩池后仍提交旧混合，实际表现一定提高吗？答：不一定。最佳安全价值不下降，实际输出可以完全不变。
- 问：新 payoff 向量离旧凸包很远，一定有用吗？答：不一定，它可能在所有参考对手上更差。
- 问：COLE 的 Shapley 思想和 COMA baseline 相同吗？答：不同。前者评价策略课程，后者构造动作梯度 baseline。
- 问：部署权重固定、recurrent state 改变，是否完全没有适应？答：仍可能适应，只是载体不同。
- 实验：固定协调矩阵，改变训练伙伴分布，再用同一冻结测试分布重算回报，检查课程分数上升是否代表泛化改善。
- 实验：复算加权剪刀石头布的 Gap 2→20；分别记录池内 Gap、完整 Gap、最新策略收益与种群安全价值，解释排序差异。



<a id="chapter-code"></a>

## 下载与运行

精确有限游戏的评价与学习目标诊断；不是神经 PSRO、COLE 或 HOLA 的论文性能复现。

[下载 marl_objectives_lab.py](../../examples/marl_objectives_lab.py)

```sh
python3 examples/marl_objectives_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Albrecht、Christianos、Schäfer · Multi-Agent Reinforcement Learning](https://www.marl-book.com/)：先读博弈与解概念，再读深度算法和实现；官网提供教材、讲义与代码。

- [Lanctot et al. · A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)：PSRO 元博弈、元策略与学习响应接口；框架和具体解概念分开。

- [Zhou et al. · Efficient Policy Space Response Oracles](https://arxiv.org/abs/2202.00633)：URR、response 与元策略联合学习、warm start 和并行；保留均衡与遗憾条件。

- [Perez-Nieves et al. · Modelling Behavioural Diversity for Learning in Open-Ended Games](https://proceedings.mlr.press/v139/perez-nieves21a.html)：DPP 期望集合大小和 payoff 几何，与动作层 Q-DPP 不同。

- [Liu et al. · Towards Unifying Behavioral and Response Diversity](https://arxiv.org/abs/2106.04958)：占据分布、响应向量、gamescape 与 population effectivity。

- [Feng et al. · Neural Auto-Curricula](https://arxiv.org/abs/2106.02745)：依据后续可利用度训练课程生成器，连接种群学习与元学习。

- [Lian et al. · Fusion-PSRO](https://doi.org/10.3233/FAIA251106)：Nash 加权参数融合用于 response 初始化，不等于概率混合。

- [Huang et al. · Conflux-PSRO](https://arxiv.org/abs/2410.22776)：状态级路由复用子策略，再蒸馏；机制说明不等于无条件性能排序。

- [McAleer et al. · XDO: A Double Oracle Algorithm for Extensive-Form Games](https://arxiv.org/abs/2103.06426)：信息状态级组合及 NXDO；相邻团队研究，表格界与神经近似分开。

- [Li et al. · Cooperative Open-ended Learning Framework for Zero-shot Coordination](https://proceedings.mlr.press/v202/li23au.html)：ICML 2023 原始 COLE：图评价、兼容性课程与局部偏好目标。

- [Li et al. · Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination](https://doi.org/10.1613/jair.1.15884)：JAIR 2024 扩展，含 COLE-SV、COLE-R 与人机研究。

- [Li et al. · HOLA-Drone](https://arxiv.org/abs/2409.08767)：超图合作关系、多伙伴课程与实体无人机零样本配合，部署策略固定。

- [Li et al. · Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales](https://arxiv.org/abs/2607.04972)：后续环境、伙伴与规模变化研究；无微调不等于部署阶段参数学习。

- [Wang et al. · ZSC-Eval](https://arxiv.org/abs/2310.05208)：测试伙伴生成、BR-Div 与 BR-Prox，评价未见伙伴泛化。

- [Yang et al. · Multi-Agent Determinantal Q-Learning](https://proceedings.mlr.press/v119/yang20i.html)：联合行动探索结构，连接但不代替种群课程。

- [Chen et al. · Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841)：固定数据上的对手重加权和离线响应；数据支持缺口不能由权重消除。

- [Liu et al. · Leveraging Team Correlation for Approximating Equilibrium in Two-Team Zero-Sum Games](https://arxiv.org/abs/2403.00255)：团队相关、偏离空间与受限团队解，连接合作内层与竞争外层。

- [Liu et al. · Computing Ex Ante Equilibrium in Heterogeneous Zero-Sum Team Games](https://arxiv.org/abs/2410.01575)：H-PSRO 的异质团队表达与顺序更新；区分局部训练界和总体均衡近似。

- [Mguni et al. · Learning in Nonzero-Sum Stochastic Games with Potentials](https://proceedings.mlr.press/v139/mguni21a.html)：特殊势结构下的一般和学习，不覆盖任意利益关系。

- [Zhou et al. · MALib](https://jmlr.org/papers/v24/22-0169.html)：JMLR 2023 种群训练系统，组织采样、评价与训练的嵌套工作负载。


<a id="marl-mechanism-experiments"></a>

## 小型实验：把更新规则与评价对象分开

全部结果来自完全列举的有限博弈，没有采样误差或置信区间。它们检验具体机制，不是大型神经算法复现。

### 末次策略与历史混合

标准石头剪刀布；从一次石头观测开始，双方精确响应历史平均策略，平局选最小索引。300轮后，末次纯策略的完整博弈gap仍为2，历史混合为0.0532。总体下降不意味着逐轮单调。

![虚拟对弈](https://yingwen.io/crl-code/marl-objectives/population.svg)

### 扩张种群的反例

收益矩阵((0,−1,1),(1,0,−10),(−1,10,0))。双方初始种群只有动作0。加入精确最佳响应动作1后，受限均衡变成双方都选1，完整博弈gap由2增至20。有限博弈DO的终止结论，不保证每轮受限均衡的full-game gap下降。

### 训练伙伴与未见伙伴

同动作得1，异动作得0。训练伙伴选0的概率为0.9。始终选0的训练回报为0.9，但面对始终选1的测试伙伴回报为0；均匀策略在两个分布下均为0.5。应固定评价分布，单独报告分布外泛化与适应成本。

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

- [探索与经验选择](../../textbook/exploration.md)
- [目标条件化与子任务构造](../../textbook/goals.md)
- [元学习与学习规则的适应](../../textbook/meta.md)
- [知识保留与再适应](../../textbook/retention.md)
- [实验设计、统计与算法测试](../../textbook/experiments.md)
- [持续学习的智能体架构](../../textbook/architectures.md)

对应原始材料：MARL book 第 3–6 章：博弈、解概念与博弈学习；MARL book 第 9–11 章：深度算法、实现与环境；PSRO、EPSRO、COLE 与 HOLA 原文。本文为原创讲解，原书、论文与上游代码保留各自许可。
