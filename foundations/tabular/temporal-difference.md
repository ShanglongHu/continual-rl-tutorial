# TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

## 本章内容

- 从 Bellman 样本推导 TD(0)。
- 区分四种控制 target、行为策略和更新顺序。
- 理解 maximization bias、Double 选择评价分离及表格收敛条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

模型未知，只收到逐步转移。既要定义预测目标，也要说明控制更新采用哪一种未来行为。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 转移 $(S_t,A_t,R_{t+1},S_{t+1})$；策略评价时固定 $\pi$，控制时可更新行为。

### 需要求解的对象

预测时估计 $v_\pi$；控制时学习动作价值并生成更好的行为。

### 信息与数据权限

每次更新只需当前转移和已有估计；没有真实的全部未来回报。

$$
v_\pi=T_\pi v_\pi,\qquad q_*=T_*q_*,\qquad \delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

前两式分别定义固定策略与最优控制的目标；TD 残差是利用一步样本构造的更新信号。残差不是外部目标的定义。

### 成立条件与解的含义

- 经典收敛结论要求平稳表格问题、充分访问以及合适步长。
- SARSA 的控制收敛还需规定探索随时间的变化；固定探索策略评价不是最优贪心控制。

判断准则：在有解析解的小 MDP 检查目标固定点，再在控制中报告实际行为收益和覆盖。

### 适用边界

- 由一次 TD loss 降低推出策略收益提高。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](../../textbook/control.md)：本章的控制部分限定为平稳充分状态和独立表格参数；一般历史依赖学习器不必满足这些条件。共享神经表示不是表格方法的前提。

- 组合不同学习问题 · [时间信用分配与资格迹](../../textbook/credit.md)：本章的一步预测与控制可接入多步或资格迹的反馈传播机制，而不重新定义外部控制目标。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整未来尚不可用；控制还可能用带噪声的估计同时选择和评价动作。

### 本章的核心思路

用后继预测自举，并明确区分下一动作采样、策略期望与最大化。

1. [用一步样本估计 Bellman 备份](temporal-difference.md#lesson-derive)：自举允许立即更新，但目标也依赖正在学习的估计。

2. [由未来行为确定控制目标](temporal-difference.md#td-control-targets)：SARSA、Expected SARSA 与 Q-learning 的差别在后继动作的处理，不只是代码写法。

3. [拆开动作选择与评价](temporal-difference.md#td-double)：Double Q 用不同估计器减弱选择噪声造成的最大化偏差。

结论与条件：经典表格保证有访问和步长条件；Double Q 缓解一种偏差，不保证所有有限样本误差更小。

### 相关方法改变了什么

- SARSA / Q-learning：分别跟踪实际下一动作与贪心最优备份。

- Expected SARSA / Double Q：前者积分掉动作采样噪声，后者分离选择和评价噪声。


<a id="lesson-setting"></a>

## 1 · 一条转移能提供什么

如果一项任务要很久才结束，等完整回报就会推迟每次改进。另一方面，智能体可能早已多次到过眼前的后继状态，对它的未来有一定认识。TD 利用这份已有预测补足尚未发生的结果：新经验只提供一步，剩余部分由当前知识提供。于是学习可以逐步发生，但旧预测的误差也进入了新目标。

模型未知，但可观察 $(S_t,A_t,R_{t+1},S_{t+1})$。先固定目标策略 π，环境有限平稳、奖励有界，使用折扣回报或适当终止任务。真实价值满足 $v_\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid s]$。

若当前后继预测等于真实价值，那么一步奖励加后继预测的条件期望就是当前真实价值。实际后继预测并不准确，TD 仍把它当作目前可用的未来估计。这种 bootstrap 让更新不必等到 episode 结束。

<a id="lesson-derive"></a>

## 2 · TD(0) 的目标与误差

$$
y_t=R_{t+1}+\gamma V_t(S_{t+1}),\quad\delta_t=y_t-V_t(S_t),\quad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

先用旧表读取当前和后继预测，再只更新当前表项。真正终止时后继项为零。δ 是单步自举误差，不是未知的真实价值误差。

这与 MC 的增量均值具有相同外形，但监督目标不同。MC 使用完整随机回报；TD 使用实际一步加已有预测。正确价值只要求条件期望中的 TD error 为零，不要求每个随机样本的误差都为零。

$$
G_t-V(S_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
$$

这条轨迹恒等式先固定 V，且终点价值为零。展开右侧后，相邻后继价值与当前价值相消；在线每步变化 V 时，需要另外保留误差项，不能原样声称严格等价。

用两步确定性轨迹 A→B→终点，奖励 0、1，γ=.9，初值零，α=.1。第一遍 TD 在 A 得到 target 0，先不变；B 更新为 .1。第二遍 A 才变成 .009。MC 第一遍则直接将 A 更新到 .09。

但“MC 第一遍传播更远”不等于 MC 在给定数据上总是更好。先分开三个问题：真实环境的价值是多少；有限数据支持哪个估计；采用什么更新顺序计算这个估计。即使使用同一批数据，MC 和 TD 也会因利用序列结构的方式不同而收敛到不同答案。

| 固定数据，γ=1，终点价值为0 | 观察到的经历 | 出现次数 |
| --- | --- | --- |
| 经历甲 | A → B → 终点；奖励依次为0、0 | 1 |
| 经历乙 | B → 终点；奖励为1 | 3 |

重复使用这四条经历。批量 MC 对每个状态的已观察完整回报做最小二乘：A 只出现一次，回报为0；B 的四次回报为0、1、1、1。因此 MC 得到 V(A)=0、V(B)=3/4。注意：这在评价有限样本回报拟合，不是已经知道真实环境中 A 的价值为0。

$$
\begin{aligned}0&=V(B)-V(A),\\0&=(0-V(B))+3(1-V(B)),\\V_{\rm TD}(A)&=V_{\rm TD}(B)=3/4.\end{aligned}
$$

批量 TD 每轮先用同一张旧表计算全部 TD 增量，再累加更新。足够小步长下，其极限让每个状态的总 TD 增量为零。A 总到 B，TD 因而利用了在其他经历中获得的 B 的信息。

还可以把数据变成经验模型：A 必到 B，B 必终止，B 的平均奖励为3/4。这个经验模型的 Bellman 解恰是上述 TD 解。这称为对经验 Markov 模型的确定性等价估计。它没有证明经验模型就是真实模型；若 A 后的 B 与直接从 B 开始的经历有不同隐藏情境，合并它们反而可能有偏。

同一有限数据集上的批量 MC 与 TD；Python 3，可独立运行

```python
returns_A = [0.0]
returns_B = [0.0, 1.0, 1.0, 1.0]
mc_A = sum(returns_A) / len(returns_A)
mc_B = sum(returns_B) / len(returns_B)
td_A = td_B = 0.0
for _ in range(2000):
    delta_A = td_B - td_A
    delta_B = sum(r - td_B for r in returns_B)
    td_A, td_B = td_A + 0.1 * delta_A, td_B + 0.1 * delta_B
assert abs(td_A - 0.75) < 1e-10
assert abs(td_B - 0.75) < 1e-10
print("MC:", mc_A, mc_B, "batch TD:", td_A, td_B)
```

这也给出通向函数逼近的关键问题。表格只在相同状态间共享后续经验；函数逼近还会把不同状态绑定到同一参数。MC 的最小二乘目标于是变成投影真实价值，TD 的平衡条件变成投影 Bellman 固定点。进入预测章时，需要比较这两个方程，而不能把所有“误差乘步长”的更新都当作同一个回归算法。

<a id="experiment-td0"></a>

### 实验：实验 · 能立即更新，不等于有限预算内必然更准确

TD 每条转移都学习，MC 必须等到结束；为什么这组 MC 曲线反而更低？

**环境与可用信息。** 五个非终止状态的等概率左右随机游走。每回合从状态 3 开始，到左端奖励 0、右端奖励 1，γ=1。状态真值依次为 1/6 到 5/6，表格估值全零。

**设置。** 五种子各运行 1200 个真实转移，固定策略使两方法可收到相同轨迹。TD(0) 每步以 α=0.1 更新当前状态。对照保存完整回合，以每状态的首次访问回报做样本平均；未结束的预算尾段不产生 MC 更新。

**检验的机制。** TD 用后继的当前估值自举，能及时更新，但零初值下正奖励需要多次访问逐步传回。MC 一旦看到终点，就能给该回合多个首次访问状态提供完整目标。这里还同时改变了固定步长与递减步长。

**测量。** 主图是五个状态相对于解析真值的等权 RMSE。横轴是真实转移，不是完整回合数，也不是总参数更新次数。MC 保存轨迹、等待终点；TD 不需要这段回合缓冲。

```bash
python3 implementations/classic/td0.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/td0/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 environment_steps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 第 300 步，TD 的平均 RMSE 为 0.2368，MC 为 0.0527；末尾为 0.0465 对 0.0267。结果说明立即更新并不自动带来更低误差，不能把更新时机与有限预算效果视为同一个概念。

**结论边界。** 不同步长规则、零初始化和短随机游走都会影响比较。没有扫描步长，也没有相同 MC 固定步长对照。这些曲线不是 Sutton 与 Barto 原书图的复现。

**继续实验。** 保留相同轨迹，给 TD 和 MC 分别扫描固定步长，再把初值改为 0.5。另记每状态访问数和回合长度，检验早期差异究竟来自等待、自举传播还是初始偏差。

[源码](../../implementations/classic/td0.py) · [逐种子记录](https://yingwen.io/crl-code/results/td0/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/td0/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/td0/curves.json)

<a id="td-control-targets"></a>

## 3 · 控制的三种下一动作

控制保存动作价值 Q，用它改善行为。SARSA 的五个量是状态、动作、奖励、下一状态、下一动作。必须先按行为选择下一动作，再使用它的 Q 更新当前动作，之后真的执行这个已选动作。

$$
y_{\rm Sarsa}=r+\gamma Q(s',a'),\qquad y_{\rm Expected}=r+\gamma\sum_a\pi(a\mid s')Q(s',a),\qquad y_{\rm Q}=r+\gamma\max_aQ(s',a)
$$

SARSA 采样下一动作；Expected SARSA 对指定目标策略求平均；Q-learning 是目标取贪心策略的特例。

Expected SARSA 可以 on-policy，也可以行为与目标不同。一步更新已经条件于当前状态动作，环境转移来自相同模型，因此直接对下一目标策略求和即可；不需要给当前样本乘一个当前动作比率来改变这个条件期望。仍然需要相关状态动作有访问覆盖。

固定 ε 的 SARSA 评价含探索的行为，Q-learning 则可一边探索一边学习贪心目标。两种 Q 数值不同不一定意味着某个实现有误。应先比较它们要评价的策略，再比较误差。

Q-learning 为什么能从探索动作学习贪心目标？在给定当前状态动作后，下一环境后果仍服从同一个转移分布。对样本 target 求条件期望，得到对后果求和、在每个后继状态选择最大 Q 的 Bellman 最优 backup。探索负责提供被更新表项的数据，而下一状态的最大值负责定义目标；二者作用不同。

<a id="td-double"></a>

## 4 · 最大化偏差与 Double Q

$$
\mathbb E[\max_a\widehat q(a)]\ge\max_a\mathbb E[\widehat q(a)]
$$

即使各动作估计无偏，最大值会偏向偶然高估的动作。两个真实值均零、估计各独立取 ±1 时，最大值的均值是 .5。

$$
a^*=\arg\max_aQ_1(s',a),\qquad y=r+\gamma Q_2(s',a^*),\qquad Q_1(s,a)\leftarrow Q_1(s,a)+\alpha[y-Q_1(s,a)]
$$

随机选择更新 Q1 或 Q2；当前表负责选动作，另一表评价。行为可依据两表之和进行 ε-greedy。

完全独立的选取误差和评价误差提供直观解释，但实际 RL 中两表共享环境与行为数据，不会保持简单独立。Double Q 缓解选择评价耦合，不保证每次无偏，也可能出现低估。它与使用延迟 target network 的 Double DQN 相联系，但不是同一个实现。

<a id="rlss-afterstate"></a>

## 动作价值之外：何时可以学习 afterstate？

动作价值把当前状态与动作作为输入。有些任务中，动作先造成一个已知的确定后果，然后世界才给出未知的响应。例如，落子后的棋盘已知，接下来对手或随机事件的响应未知。此时可以把已知部分从学习问题中分离，学习动作后状态（afterstate）的价值。它不是把任意下一观察换一个名字。

$$
\widetilde S_t=f(S_t,A_t),\qquad (R_{t+1},S_{t+1})\sim P(\cdot,\cdot\mid\widetilde S_t),\qquad q_\pi(s,a)=u_\pi(f(s,a)).
$$

本节将一次完整的动作—世界响应计作一步，奖励全部在响应时记账。已知映射 f 必须保留决定后续分布的信息；若奖励还含只依赖原状态动作的已知代价，应另外加上，不能丢失。

$$
u_\pi(\widetilde s)=\mathbb E\!\left[R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})u_\pi(f(S_{t+1},a))\mid\widetilde S_t=\widetilde s\right].
$$

对世界响应求期望，再对下一动作求期望。这是一次完整决策周期的 Bellman 方程；已知映射 f 本身不额外消耗一步折扣。

如果两对状态动作导向同一个充分的 afterstate，它们共享同一个待学习答案。一般的特征共享是一种近似；这里的合并在所列条件下是精确的。比如两个落子顺序产生同一局面、同一行动方且具有同样的规则状态，后续经验可复用。重复局面规则、已消耗资源或隐藏信息不同，则不能仅凭可见棋盘相同就合并。

**算法：Afterstate TD 的时序；已知动作后果与未知环境响应分开**

1. 先枚举合法动作，计算每个动作的已知 afterstate。
1. 按 afterstate 价值选择动作，并保存实际选择的 afterstate。
1. 执行动作，读取完整世界响应、奖励和下一个决策状态。
1. 若真实终止，target = reward。
1. 否则，按目标策略求下一 afterstate 价值的期望，或在控制中取最大值。
1. 只更新本次选择的 afterstate 价值。
1. 在后续步骤中继续探索，并独立检查充分性假设。

手算：四个动作的已知后果依次为 $(x,x,y,y)$。从 x 出发一步结束，奖励为零或二、各半；从 y 出发总获 0.6。于是 $u(x)=1,u(y)=0.6$。Q 表有四个条目，afterstate 表只需两个。观察第一个动作的后果能更新第二个动作的估值，因为两者对应同一个预测问题。

| 表示 | 共享依据 | 需要检验的条件 |
| --- | --- | --- |
| Q(s,a) | 逐状态动作保存 | 状态足以描述条件后果 |
| Afterstate u(f(s,a)) | 已知动作后果完全等价 | 给定 afterstate 后，原状态动作不再补充后果信息 |
| 神经特征上的 Q | 学习器决定的近似相似性 | 不同样本共享梯度，但目标未必相同 |

连接持续学习：afterstate 可以节省样本，却把一部分结构交给了设计者。若动作效果逐渐改变，已知 f 的假设可能失效；若 afterstate 本身要从经验学得，就同时引入状态构造和模型误差。不能把这种情况下的数据合并仍当作无损压缩。

<a id="td-loop"></a>

## 5 · 完整控制循环与终止处理

**算法：算法伪代码**

1. 初始化 Q（Double 时初始化两张表），并选择行为探索参数。
1. 每次 episode：
  1. 观察初始状态，按当前行为选择动作。
  1. 执行动作，获得奖励和下一状态。
  1. 若真实终止：target 仅为奖励，不查询下一动作。
  1. 否则：先按旧表选择下一动作，并计算算法规定的余项。
  1. 更新当前状态动作；Double 随机只更新一张表。
  1. 转到下一状态，并保留已经选择的下一动作，继续交互。

不能在形成 SARSA target 后又重新采样实际下一动作，否则记录中的 target 与真实行为序列不再按所述方式对应。预算截断也不能不加区分地当成真实终止。如果训练后想评价纯贪心策略，应在独立评测中明确关闭探索。

<a id="lesson-example"></a>

## 6 · 手算和可运行实现

| 下一状态 Q=(2,0)，r=0，γ=.9 | 规则 | target |
| --- | --- | --- |
| 实际选择第二动作 | SARSA | 0 |
| 目标概率 (.9,.1) | Expected SARSA | 1.62 |
| 贪心目标 | Q-learning | 1.8 |
| Q1=(5,4)，Q2=(1,3) | Q1选第一个，Q2评价 | 0.9；直接 max Q2 则为2.7 |

环境 A 退出得 .2，或零奖励到 B；B 得 1 或 −1 后终止。ε=.1、γ=.9，Expected SARSA 的 A继续值约 .81，Q-learning 和 Double 约 .9；固定步长 SARSA 在目标附近波动，默认末值约 .868319。这个数不是其精确期望，不能要求它逐种子等于 .81。

<a id="lesson-code"></a>

## 7 · 源码检查与改动实验

真实控制环境、已选下一动作的保留、Expected target，以及随机交换角色的 Double Q。

```python
def td_prediction(episodes=1000, alpha=0.1):
    values = [0., 0.]  # 0 --r0--> 1 --r1--> terminal
    for _ in range(episodes):
        values[0] += alpha*(.9*values[1]-values[0])
        values[1] += alpha*(1.-values[1])
    return values

def control_target(kind, reward, next_values, probabilities=None,
                   next_action=None, gamma=0.9):
    if next_values is None:
        return reward
    if kind == "sarsa":
        tail = next_values[next_action]
    elif kind == "expected":
        tail = sum(p*v for p, v in zip(probabilities, next_values))
    elif kind == "q":
        tail = max(next_values)
    else:
        raise ValueError(kind)
    return reward+gamma*tail

def td_control(kind, episodes=6000, seed=7, alpha=0.1, epsilon=0.1):
    rng = random.Random(seed)
    q1 = {s: [0.]*len(CONTROL_MODEL[s]) for s in CONTROL_MODEL}
    q2 = {s: [0.]*len(q1[s]) for s in q1}
    for _ in range(episodes):
        state = 0
        action = choose(epsilon_probs([x+y for x, y in zip(q1[state], q2[state])],
                                      epsilon), rng)
        while state is not None:
            reward, sp = step(CONTROL_MODEL, state, action, rng)
            next_action, probabilities = None, None
            if sp is not None:
                probabilities = epsilon_probs([x+y for x, y in zip(q1[sp], q2[sp])],
                                              epsilon)
                next_action = choose(probabilities, rng)
            if kind == "double":
                update, other = (q1, q2) if rng.random()<.5 else (q2, q1)
                target = reward
                if sp is not None:
                    selected = max(range(len(update[sp])), key=lambda a: update[sp][a])
                    target += .9*other[sp][selected]
                update[state][action] += alpha*(target-update[state][action])
            else:
                target = control_target(kind, reward, None if sp is None else q1[sp],
                                        probabilities, next_action)
                q1[state][action] += alpha*(target-q1[state][action])
            # Keep SARSA's already-selected next action.
            state, action = sp, next_action
    return {s: [(x+y)/2 if kind=="double" else x for x, y in zip(q1[s], q2[s])]
            for s in q1}
```

td_prediction 实现固定两步策略的预测，td_control 实现选动作、执行环境、构造 target、更新和继续行动的完整控制循环。前者只验证价值传播，不拿它的误差曲线代替控制收益。所有控制方法使用同一个奖励定义；Double 返回两表平均，便于和单表的价值尺度比较，行为选择则使用两表之和。

把步长从 .1 降到 .01，观察随机 SARSA 的波动与传播速度同时变化；再令 ε=0，检查是否因为初始并列选择而长期不去 B。后一个失败是覆盖问题，不能仅靠更小 TD error 诊断。若要研究 Double 的最大化偏差，需要额外设置多动作随机后果并跨独立运行统计；本章两步确定性环境主要检查更新角色和目标。

<a id="lesson-branches"></a>

## 8 · 收敛条件与 CRL 衔接

$$
\sum_n\alpha_n(s,a)=\infty,\qquad\sum_n\alpha_n(s,a)^2<\infty
$$

经典表格随机逼近结果使用逐状态动作的步长条件，还需要固定环境、相应访问与噪声条件。不能只检查一个全局步长序列。

SARSA 向最优策略收敛的典型条件还包括贪心极限且持续探索。固定 ε 则保留行为探索；固定步长在随机环境中通常持续波动。持续任务可能正需要固定步长追踪变化，但那是另一种估计目标，不能把固定点收敛定理直接搬过来。

函数逼近、bootstrap 和 off-policy 的组合可能不稳定，表格结论不自动扩展到共享网络。Afterstate 则是另一种表示选择：若动作先确定一个中间状态，之后才有环境随机性，可以评价这个中间状态；它不等于去掉探索或改变长期目标。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：训练 TD loss 不为零是否一定没学对？答：不一定。随机后果可以产生不可约的单样本误差，应检查条件期望、解析价值或独立评估。

问：Expected SARSA 是否天然只能 on-policy？答：不是。它对指定目标策略求和，行为可以不同；必须保持覆盖，并明确目标随训练怎样变化。

问：Double Q 为何不总是更新两张表？答：随机分别更新有助于分离选择与评价。如果两表初始化相同并每次进行完全相同更新，它们会相等，不能形成预期的分离。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](../../examples/tabular_textbook_lab.py)

```sh
python3 examples/tabular_textbook_lab.py temporal-difference
python3 examples/tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [Watkins & Dayan · Q-learning](https://doi.org/10.1007/BF00992698)：表格 Q-learning 与收敛条件。

- [van Hasselt · Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)：最大化偏差、双估计器及原始算法。

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
- [深度价值学习](../../textbook/deep-value.md)

对应原始材料：6.1–6.8。本文为原创讲解，原书、论文与上游代码保留各自许可。
