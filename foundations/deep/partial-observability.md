# 不完全可观测：信念状态、信息行动与递归记忆

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

## 本章内容

- 从历史条件分布推导 Bayes filter。
- 把信息获取的价值纳入 Bellman 决策。
- 区分精确信念、学习的 recurrent state 和训练时的隐状态权限。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

当前观测不能确定未来。智能体必须利用历史形成足够的决策信息。

### 给定条件与符号

- 有限潜在状态集 $\mathcal X$、观测集 $\mathcal O$、动作集 $\mathcal A$、奖励域 $\mathcal R\subset\mathbb R$；$X_t,O_t,A_t,R_{t+1}$ 为相应变量。平稳联合核为 $K(x',B_o,B_r\mid x,a)=\Pr(X_{t+1}=x',O_{t+1}\in B_o,R_{t+1}\in B_r\mid X_t=x,A_t=a)$，其中 $B_o,B_r$ 为可测观测与奖励事件；离散情形可写为 $K(x',o,r\mid x,a)$。
- 历史 $H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t)$；给定初始观测律 $\nu_0$ 与条件初始信念 $b_0(x\mid O_0)$，它们共同指定 $(X_0,O_0)$ 的联合分布。奖励有界，$0\le\gamma<1$。
- $\Pi_H$ 是非空的允许可测因果策略类，$\pi_t(\cdot\mid H_t)$ 只能使用已到达的信息。已知 $K,b_0$ 时可维护 $b_t(x)=\Pr(X_t=x\mid H_t)$；未知模型需另给模型族和信息条件，联合估计模型与状态，或学习近似摘要。

### 需要求解的对象

在 $\Pi_H$ 中求使折扣收益接近上确界的策略，并构造支持它的信息状态；不假定任意受限策略类的最优者必然存在。

### 信息与数据权限

执行时不可访问潜在真状态；精确 filter 的参照问题允许查询正确模型与初始律。未知模型版本不提供这项权限，须声明模型族、先验或识别数据。奖励如含状态信息，也必须纳入历史和后验。

$$
J(\pi)=\mathbb E_{\nu_0,b_0,K,\pi}\!\left[\sum_{t\ge0}\gamma^tR_{t+1}\right],\qquad J(\hat\pi)\ge\sup_{\pi\in\Pi_H}J(\pi)-\varepsilon,\quad\varepsilon>0
$$

这是指定初始化与策略类下的历史条件控制，$A_t\sim\pi_t(\cdot\mid H_t)$。正确已知模型下 belief 对未来预测充分；若策略类另有限制，其信息或资源约束也须保留，不能自动换成任意 belief 策略。

### 成立条件与解的含义

- 潜在 Markov 条件是给定 $X_t,A_t$ 后，下一潜在状态、观测和奖励的联合条件律不再依赖更早历史，并由同一 $K$ 给出。精确 filter 使用正确模型与初始律；离散观测的证据概率为正，连续情形使用相应密度或正规条件分布。
- 本章正文的分解式 filter 另假定奖励不提供额外状态信息；否则必须将奖励纳入联合似然。未知模型可在指定先验下维护状态与模型参数的联合后验，并非只能使用近似记忆。
- 有限记忆及计算限制应进入 $\Pi_H$ 或完整实现集合；近似摘要的控制损失需另检验。

判断准则：除状态预测误差，还检查不同历史被合并后是否仍能选择正确动作。

### 适用边界

- 以记忆长度或重构准确率直接证明控制充分性。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](../../textbook/state.md)：本章将隐藏 Markov 模型的 belief 构造接到历史控制；一般状态构造章已包含部分可观测过程，本章不是再次放宽它的假设。

- 组合不同学习问题 · [探索与经验选择](../../textbook/exploration.md)：本章的信息状态控制可与探索结合：动作可能通过改善信息而有价值，即使即时奖励较低。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同观测可能来自需要不同动作的情境，反应式策略无法区分。

### 本章的核心思路

先用已知模型展示准确后验如何递推，再定位学习记忆的近似位置。

1. [明确更新前后有什么信息](partial-observability.md#lesson-notation)：动作先改变潜在状态，新观测和奖励再修正后验。

2. [把历史压缩为 belief](partial-observability.md#lesson-derive)：Bayes 递推给出充分状态的参照，而非要求神经网络储存完整历史。

3. [用决策差异评价信息价值](partial-observability.md#lesson-information)：更准确识别状态只有改变未来行动或预测时才产生对应用途。

结论与条件：belief 充分性有正确模型条件；近似 recurrent state 必须通过下游任务检验。

### 相关方法改变了什么

- Bayes filter / recurrent network：前者依据指定生成模型精确递推，后者从目标和数据学习摘要。

- 观测重构 / 控制充分性：优化的误差对象不同；重构很好仍可能遗漏决策关键变量。


<a id="lesson-setting"></a>

## 1 · 观察不等于状态

前面的 Bellman 递推以状态为条件。神经网络接收一幅图像，却不等于已经获得这样的状态：同样的位置可以对应向左或向右的速度，眼前看不见的线索也可能决定下一步奖励。增加网络容量只扩大当前输入的函数类；要利用过去信息，还需说明历史怎样进入计算。这使我们回到控制问题的信息条件。

设环境隐状态为 $S_t$，智能体收到 $O_t$。同一幅图像可能来自不同速度、物体遮挡或任务阶段。即使环境隐状态满足 Markov 性，观察序列也未必满足。POMDP 是信息条件；表格 belief、线性滤波器和深度 recurrent 网络都可以处理它。它不是深度 RL 之后才出现的阶段。

$$
H_t=(O_0,A_0,R_1,\ldots,A_{t-1},R_t,O_t),\qquad b_t(s)=\Pr(S_t=s\mid H_t)
$$

历史是已经到达的信息；belief 是给定模型时对当前隐状态的后验分布，不是对网络参数的置信区间。

以下有限状态推导假定模型已知。先讨论奖励没有提供额外状态信息的情形，再用联合转移—奖励—观测模型纳入奖励信息。设计者提供了状态假设、传感器接口和模型结构；运行时智能体更新 belief 并选择行动。

<a id="lesson-notation"></a>

## 2 · 动作之后，先预测再条件化

$$
T_a(s,s')=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a),\qquad O_a(o\mid s')=\Pr(O_{t+1}=o\mid S_{t+1}=s',A_t=a)
$$

转移矩阵每一行和为一。观测似然按收到的观测取值，它不是在状态维度归一化的 posterior。

$$
\bar b_{t+1}(s')=\sum_sT_a(s,s')b_t(s)
$$

用全概率公式将旧状态不确定性传播到下一状态。这一步尚未使用新观察。

$$
\Pr(o\mid b_t,a)=\sum_{s'}O_a(o\mid s')\bar b_{t+1}(s')
$$

这是新观测的证据概率，随后用作归一化常数。

<a id="lesson-derive"></a>

## 3 · Bayes 递推与 belief MDP

$$
b_{t+1}(s')=\frac{O_a(o\mid s')\sum_sT_a(s,s')b_t(s)}{\sum_{\tilde s}O_a(o\mid\tilde s)\sum_sT_a(s,\tilde s)b_t(s)}=:B(b_t,a,o)(s')
$$

分子是下一状态与观测的联合条件概率，分母消去状态得到观测概率。分母为零时模型认为该观测不可能，不能静默归一化。

$$
b_{t+1}(s')=\frac{\sum_s b_t(s)K(s',r,o\mid s,a)}{\sum_{\tilde s,s}b_t(s)K(\tilde s,r,o\mid s,a)}=:B_K(b_t,a,r,o)(s')
$$

奖励携带信息时，K 是后继状态、已收到奖励与观察的联合条件核。这里为便于求和取离散变量；连续变量需用相应密度。这一形式保留奖励、观察与转移之间的相关性。

例如隐藏状态在本步保持不变，普通观察完全相同，奖励却等于隐藏的二元状态。先验各半，收到奖励一后正确后验把全部质量放在状态一；忽略奖励的 filter 仍是各半。奖励既是目标信号，也可能是可用信息，二者不能只保留其一。

给定正确模型与初始分布，belief 对未来控制是充分的：已知当前 belief 和新动作，即可预测后继 belief 及奖励，不必再次读取整段历史。状态集合变成概率单纯形，通常连续，即使原隐状态只有有限个。充分性来自模型和条件分布，不来自把向量命名为 state。

$$
\Pr(S_{t+1}=s',R_{t+1}=r,O_{t+1}=o\mid H_t,a)=\sum_s b_t(s)K(s',r,o\mid s,a)
$$

先对当前隐藏状态用全概率公式求和。两个历史若给出相同 belief，对任何新动作便给出相同的后果分布；相同后果再经 Bayes 更新给出相同后验。逐步重复这两个事实，才说明 belief 保留了未来预测所需的历史信息。这里假设世界模型固定且正确。

$$
\Pr(r,o\mid b,a)=\sum_{s,s'}b(s)K(s',r,o\mid s,a),\qquad \bar r(b,a)=\sum_{r,o}r\Pr(r,o\mid b,a)
$$

联合后果概率既给出即时奖励的条件均值，也给出不同后验 belief 的发生概率。

$$
V^*(b)=\max_a\left[\bar r(b,a)+\gamma\sum_{r,o}\Pr(r,o\mid b,a)V^*(B_K(b,a,r,o))\right]
$$

行动既影响外部状态，也影响未来信息。奖励携带的信息必须进入后继 belief，不能只把奖励换成均值。若给定动作与观察后奖励没有额外状态信息，$B_K$ 不再依赖 r，才可把后续项化为仅对 o 求和的 $B(b,a,o)$ 形式。

<a id="lesson-information"></a>

## 4 · 信息只有在改变决策时才有控制价值

$$
\operatorname{VOI}=\sum_o\Pr(o\mid b)\max_a\sum_sb(s\mid o)r(s,a)-\max_a\sum_sb(s)r(s,a)-c_{\rm sense}
$$

这是静态隐状态、先感知一次再作终端决策的价值差；不是所有序列任务的一般信息价值公式。

若不计感知成本，观察后仍可采用原动作，因此这里的信息价值非负。加入成本后可能为负。减少 belief 的熵不必改善控制：传感器可能准确识别与奖励无关的细节。控制充分的状态只需保留会影响最优选择的信息，不必重建全部原始观察。

对模型未知的任务，还需区分“当前状态在哪里”与“动力学是什么”两种不确定性。仅对状态做 Bayes filtering，并没有自动学习未知转移。将模型参数也作为隐变量会得到更大的信念空间，计算成本随之增加。

<a id="experiment-extended-bayes_filter"></a>

### 实验：实验 · 一次错误观测，应当推翻过去的全部证据吗？

当隐藏状态通常保持不变、观测偶尔出错时，递归信念能否比只看当前观测更准确？

**环境与可用信息。** 两个隐藏状态以 0.9 的概率保持、0.1 的概率切换。观测以 0.8 的概率报告正确状态。学习器知道这两个概率，只看到观测；真实状态仅供评价使用。初始信念为 [0.5,0.5]。

**设置。** 种子 0–4，各处理 1200 条观测。Bayes filter 先传播旧信念，再用似然校正。无记忆对照每步都从均匀先验开始。两者接收相同种子的同一隐藏状态与观测流。没有梯度训练，也没有控制动作。

**检验的机制。** 变化只在于是否保留上一时刻的概率分布。连续一致的观测会积累证据；孤立的相反观测不必立即翻转判断。状态真的切换时，旧信念又会使反应产生滞后。

**测量。** 纵轴是观测后给真实隐藏状态分配的负对数概率，采用 0.95×旧值 + 0.05×新损失的滑动平均。越低越好。它评价概率质量，不是仅评价最可能状态是否猜对，也不是 RL 回报。

```bash
python3 implementations/extended_adaptation/bayes_filter.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-bayes_filter/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 observations。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 第 1200 条观测后，五种子的平均损失为 0.353 nats，无记忆对照为 0.478 nats。记忆在该持久状态模型中有帮助，但曲线仍随错误观测与真实切换波动；滤波不是把不确定性消除为零。

**结论边界。** 这是已知正确模型的状态估计，不包含未知模型学习或策略改善。图中的末值是近期损失，不是全程平均。若实际切换率与假定模型不符，旧信念也可能有害。

**继续实验。** 先手算连续三次相同观测后的信念，再接入一次相反观测。然后分别只改变真实切换率与滤波器假定切换率，比较“需要更强记忆”和“模型失配”两种解释。

[源码](../../implementations/extended_adaptation/bayes_filter.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-bayes_filter/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-bayes_filter/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-bayes_filter/curves.json)

<a id="lesson-algorithm"></a>

## 5 · 精确 filter 与学习的 recurrent state

**算法：外部 reset 改变隐状态分布时，需要相应重置先验；普通训练 batch 边界不自动改变世界。**

1. 初始化已声明的先验 belief。
1. 根据当前 belief 计算或近似价值，选择动作。
1. 取得下一观察与奖励；仅使用已经到达的数据。
1. 用联合模型计算下一状态及实际奖励、观察的概率。
1. 按已收到的奖励与观察条件化，检查证据概率，再归一化。
1. 将后验保存为下一步 belief；重复行动。

$$
z_{t+1}=f_\theta(z_t,A_t,O_{t+1},R_{t+1}),\qquad \hat y_{t+1}=g_\theta(z_{t+1})
$$

RNN 用可训练的有限维摘要替代显式 belief；预测头或控制损失提供训练信号。

RNN 能携带历史，但不因此等于精确后验。预测下一像素、预测多种未来信号和优化控制回报会保留不同信息。TBPTT 截断参数梯度，hidden-state reset 则清除记忆内容；即使二者都发生在同一个代码边界，它们仍是不同操作。

训练时使用真实隐状态的 critic 也要声明评价对象。若 actor 带记忆，同一个环境状态搭配不同的 actor 记忆，未来行为可以不同；只给 critic 环境状态未必足以定义一个固定的 Bellman 递推。充分的集中上下文可包括隐状态与必要的控制器记忆，actor 执行权限仍保持原来的观察历史。

<a id="course-recurrent-credit"></a>

## 5.1 · 保存历史信息不等于学会怎样保存

给定递归状态 $h_t=f_\theta(h_{t-1},o_t,a_{t-1})$，其数值负责把过去信息带到当前。另一个对象是损失对过去计算的梯度。为了写清楚，先假设整段历史由同一组固定参数计算，再令 $P_t=\partial h_t/\partial\theta$。

$$
P_t=\frac{\partial f_\theta}{\partial\theta}
+\frac{\partial f_\theta}{\partial h_{t-1}}P_{t-1}
$$

第一项是当前计算对参数的直接依赖；第二项把过去依赖传播过来。BPTT 反向展开这张计算图；RTRL 前向递推敏感度。两者的时间顺序和存储代价不同。

例如 $h_t=0.99h_{t-1}+o_t$。一百步前的观察对当前数值的系数仍为 $0.99^{100}\approx0.366$。若只保留最近三十步的反向图，早于该边界的计算被当作常数，其参数梯度路径已经断开。hidden state 没有清零，不等于百步前的“如何写入记忆”仍能获得梯度。

把旧 hidden state 存入 replay，又出现另一个问题：这个向量由旧 encoder 生成。当前网络可能已经重新分配其坐标含义。用当前参数对一段旧观察做 burn-in 能缓解失配，但 burn-in 有限，未必恢复远期信息；从零状态开始则直接放弃此前历史。R2D2 专门研究了这种表示漂移与状态陈旧性。

参数逐步改变时，还需声明要对哪个计算过程求导：固定某组参数重新解释历史，还是对实际用多组参数执行过的学习轨迹求导。上面的共享参数敏感度递推对应前者的计算图，不能直接冒充“整个在线学习过程”的精确梯度。资格迹与近似在线梯度分别提供不同的折中。

与持续学习的连接：在一个长期运行、不方便重置的世界里，清空 hidden state、截断梯度、清空 replay 和重置参数是四个不同的操作。报告中必须写出实际发生了哪个操作，才能判断算法是否真的保留并利用长时经验。

<a id="lesson-example"></a>

## 6 · 两个可手算的例子

先验为 $(0.6,0.4)$，转移矩阵两行为 $(0.9,0.1)$ 与 $(0.2,0.8)$。预测分布是 $(0.62,0.38)$。收到观测的似然为 $(0.8,0.2)$，未归一化后验为 $(0.496,0.076)$，证据为 $0.572$，后验约为 $(0.86713,0.13287)$。直接将似然归一化为 $(0.8,0.2)$ 会丢掉先验和动态预测。

另一个任务只有左右两扇门，隐藏的正确门先验各半，选对得一、选错得负一。立即选择的期望收益为零。准确率 0.8 的传感器让观察后的最佳选择收益为 0.6；感知成本 0.1，净收益为 0.5。若传感器完全无信息，净增益为 −0.1。

<a id="lesson-code"></a>

## 7 · 可运行 filter 与信息价值

矩阵预测、Bayes 归一化与一次感知的期望决策价值；两个例子不依赖采样误差。

```python
def belief_update(prior, transition, likelihood):
    """Transition[s][s_next], likelihood[s_next] for the observation received."""
    probabilities(prior)
    n = len(prior)
    if len(transition) != n or len(likelihood) != n or any(len(row) != n for row in transition):
        raise ValueError('matching state dimensions required')
    for row in transition:
        probabilities(row)
    if any(not 0 <= p <= 1 for p in likelihood):
        raise ValueError('observation likelihood must be in [0,1]')
    predicted = [sum(prior[s]*transition[s][sp] for s in range(n)) for sp in range(n)]
    evidence = dot(predicted, likelihood)
    if evidence <= 0:
        raise ValueError('impossible observation under this model')
    return [p*l/evidence for p, l in zip(predicted, likelihood)], evidence


def information_value(prior, sensor, rewards, sensing_cost=0.):
    """One sensing step, then choose one terminal action.

    sensor[state][observation]; rewards[action][state]; hidden state is static.
    """
    probabilities(prior)
    for row in sensor:
        probabilities(row)
    before = max(dot(prior, r) for r in rewards)
    after = 0.
    for o in range(len(sensor[0])):
        joint = [p*row[o] for p, row in zip(prior, sensor)]
        evidence = sum(joint)
        if evidence:
            posterior = [x/evidence for x in joint]
            after += evidence*max(dot(posterior, r) for r in rewards)
    return before, after-sensing_cost, after-sensing_cost-before
```

执行 python3 examples/extended_foundations_lab.py demo 查看 posterior、evidence 和信息收益；执行 test 检查归一化、零证据与无信息传感器。代码没有训练 recurrent 网络，也没有实现一般 POMDP 规划器。pomdp-solve 是 Cassandra 的经典求解软件；pomdp-py 是后续研究框架，二者的归属与实现范围不同。

<a id="lesson-branches"></a>

## 8 · 假设边界与持续学习接口

- 模型错误：精确计算错误模型下的 posterior，仍可能产生系统性误判。
- 信息泄漏：训练时的 simulator state 可以帮助 critic，但执行策略若依赖它，就不是相同观察条件。
- 序列失配：从 replay 抽单帧无法一般性重建 history-dependent state；burn-in 也需要明确参数版本。
- 变化环境：原观测或转移模型失效时，belief 可能越来越自信地错误，而不是自动适应。

与 CRL 的连接是状态持续构建和状态模型持续校准。可固定任务奖励，单独改变传感器可靠度，比较已知新模型的 Bayes oracle、旧模型 filter 和学习的 recurrent state。这样能区分信息不足、模型滞后与优化失败，不能仅凭回报下降判断遗忘。

<a id="rlss-hidden-state-bootstrap"></a>

## 不充分的状态下，Monte Carlo、TD 与资格迹分别能修复什么？

先固定策略，只考虑价值预测。设真实过程是有限 Markov 奖励过程，但学习器只看到其特征。不同真实状态可以产生同一个特征。此时有两个独立困难：表示无法区分不同未来；自举又把这种不准确的预测当成学习目标。更长的回报可以减轻第二项，却不能凭空恢复第一项丢失的信息。

$$
v=(I-\gamma P)^{-1}r,\qquad \widehat v=\Phi w,\qquad
\Pi_Dv=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top Dv.
$$

$P$ 是固定策略的转移矩阵，$r$ 是一步期望奖励，$0\leq\gamma<1$。$D$ 是访问权重的对角矩阵；假设所有相关状态有正权重，且 $\Phi$ 列满秩。右式是这个表示在加权平方误差下能达到的最佳预测，不是每个隐藏状态的真价值。

若两个等频出现的隐藏状态有价值 2 和 8，而特征完全相同，最佳单一预测是 5，均方误差仍为 9。更多样本能更准确地学到 5，却不能把两个情形分开。MC 学习的是给定现有信息的平均回报；它没有偷偷获得真实状态。

$$
\begin{aligned}
 A_\lambda&=\Phi^\top D(I-\gamma\lambda P)^{-1}(I-\gamma P)\Phi,\\
 b_\lambda&=\Phi^\top D(I-\gamma\lambda P)^{-1}r,\qquad A_\lambda w_\lambda=b_\lambda.
 \end{aligned}
$$

这是固定特征、固定策略、相应访问分布下线性 TD(λ) 的期望固定点方程。由多步 Bellman 算子的几何级数得到；它不是非线性网络、变化策略或任意数据分布的稳定性声明。

$$
\lambda=1:\quad A_1=\Phi^\top D\Phi,\qquad b_1=\Phi^\top Dv,\qquad \Phi w_1=\Pi_Dv.
$$

这里反矩阵相消。λ 趋向一时，期望固定点趋向直接回报回归的投影。实际采样仍有方差、回报延迟与有限步长误差。

对持续、平稳、on-policy 的链，取 $D$ 为其稳态分布。投影后的多步算子的一个压缩系数是 $\kappa_\lambda=\gamma(1-\lambda)/(1-\gamma\lambda)$。三角不等式给出下面这个保守的范数界。它解释趋势，不断言某个有限样本实验随 λ 单调改善。

$$
\|\Phi w_\lambda-v\|_D\leq
 \frac{1}{1-\kappa_\lambda}\|\Pi_Dv-v\|_D
 =\frac{1-\gamma\lambda}{1-\gamma}\|\Pi_Dv-v\|_D.
$$

把 v 与其投影插入固定点误差，并把压缩项移到左边即可得到。回合访问权重、离策略权重或随时间变化的特征不能不加检查地套用这组假设。

固定点诊断：区分表示误差与自举引入的偏离；不是训练曲线。

```python
import numpy as np
P = np.array([[.8, .2, 0.], [0., .5, .5], [.3, 0., .7]])
r = np.array([1., 0., -1.])
Phi = np.array([[1., 0.], [1., 0.], [0., 1.]])
gamma = .9
# 先求稳态分布；这个诊断拥有真实模型，在线学习器并没有。
d = np.linalg.lstsq(np.vstack([P.T-np.eye(3), np.ones(3)]),
                    np.r_[np.zeros(3), 1.], rcond=None)[0]
D = np.diag(d)
v = np.linalg.solve(np.eye(3)-gamma*P, r)
projected = Phi @ np.linalg.solve(Phi.T@D@Phi, Phi.T@D@v)
for lam in (0., .5, 1.):
    M = np.linalg.solve(np.eye(3)-gamma*lam*P, np.eye(3))
    w = np.linalg.solve(Phi.T@D@M@(np.eye(3)-gamma*P)@Phi,
                        Phi.T@D@M@r)
    print(lam, np.sqrt(d @ (Phi@w-v)**2))
    if lam == 1.: assert np.allclose(Phi@w, projected)
```

课程中的 bit-to-bit 练习则走向另一个问题：行动会影响获得的信息。增加历史特征、构造可递归更新的状态、选择能区分假设的行动，才可能减少信息混叠。长观察窗口也会增加稀疏度和估计难度。比较状态构造时，应给它们相同的数据与存储预算；不能把拥有真实隐藏状态的零表示误差与有限样本学习误差直接比较。

接入控制后还要重新检查策略类。某个固定策略下两个历史的回报相同，并不表示对所有未来行动都相同。价值预测器可以在当前策略上工作良好，但规划器仍需要区分这些历史。状态、预测问题与控制权限必须共同声明。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：观测看起来相同，是否必须采取相同动作？答：不必；不同历史可导致不同 belief。
- 问：belief 熵下降是否意味着决策价值增加？答：不保证。它可能只减少无关信息的不确定性，且感知有成本。
- 问：Bayes 更新的证据为零时可以加一个很小的数继续吗？答：数值平滑可以作为建模改动，但不能隐去模型与观测矛盾；需要报告所用平滑和支持假设。
- 实验：将传感器准确率从 0.8 改为 0.5。在门任务里后验保持先验，信息净收益应为 −0.1；此结论不依赖控制网络。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](../../examples/extended_foundations_lab.py)

```sh
python3 examples/extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Kaelbling、Littman、Cassandra · Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf)：作者托管原文；belief、决策与 POMDP 求解。

- [Cassandra · pomdp-solve](https://www.pomdp.org/code/index.html)：经典 POMDP 求解软件的作者入口，并非神经 recurrent agent。

- [h2r · pomdp-py](https://github.com/h2r/pomdp-py)：框架作者的模型、belief 与规划接口；是后续工具，不是 1998 原文代码。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

- [Kapturowski et al. · Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)：原论文分析 replay 中的参数延迟、表示漂移、recurrent state staleness，以及 stored state 和 burn-in 的取舍。

- [Baisero & Amato · Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf)：§4–5：state-only critic 的条件与 history-state value；额外训练信息不自动消除历史依赖。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 当前输入保留了哪些历史信息？

Bellman 方程先假定有足够的状态。表格为不同状态分别存值；它不负责从相同观察中恢复被遗漏的历史。

函数逼近与深度方法：特征共享是在已知信息上泛化；递归状态则保留过去信息。增加网络宽度不等于补回历史，低训练误差也不证明输入满足 Markov 性。

持续学习中的研究问题：策略改变以后，原来的状态压缩是否仍能预测行动后果？构造状态的网络、运行时记忆、资格迹与优化器状态如何共同更新？

[MDP 的状态条件](../tabular/mdps.md) → [表示与泛化](../approximation/features-control.md) → [不完全可观测](partial-observability.md) → [智能体状态](../../textbook/state.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [智能体状态与递归学习](../../textbook/state.md)
- [通用价值函数与预测知识](../../textbook/gvf.md)
- [时间信用分配与资格迹](../../textbook/credit.md)
- [转移模型与后果模型](../../textbook/models.md)

对应原始材料：Kaelbling、Littman、Cassandra：POMDP；Algorithms for Decision Making：state uncertainty。本文为原创讲解，原书、论文与上游代码保留各自许可。
