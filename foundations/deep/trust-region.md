# 策略更新的尺度：TRPO 与 PPO

同一批数据可以重复使用多少次？为什么 clipping 不是一个性能保证？

## 本章内容

- 由性能差分恒等式解释 surrogate 的来源。
- 求解局部 KL 约束，理解 Fisher、共轭梯度与回溯。
- 按优势符号解释 PPO，并固定旧策略、优势与 critic target。

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

已有旧策略生成的一批轨迹，希望在有限数据下改变策略，又不让分布变化破坏局部近似。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 旧策略 $\pi_{\rm old}$、其折扣占用分布 $d_{\rm old}$、优势估计 $\hat A$、KL 预算 $\delta>0$。

### 需要求解的对象

选择局部策略更新；真实外部目标是 J，局部 surrogate 只是可计算替代。

### 信息与数据权限

更新时不能立即知道新策略的完整占用分布；重复使用同批数据会逐渐偏离采样策略。

$$
\max_\theta\mathbb E_{s\sim d_{\rm old},a\sim\pi_{\rm old}}\!\left[\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}\hat A(s,a)\right]\quad\text{s.t.}\quad\mathbb E_{d_{\rm old}}[D_{\rm KL}(\pi_{\rm old}\|\pi_\theta)]\le\delta
$$

这是常用平均 KL 信赖域近似。严谨性能界中的最坏状态分布偏差、优势误差与实践平均约束不能混为一谈。

### 成立条件与解的含义

- 旧策略支持需覆盖所评价动作；批内旧 log-prob 和优势版本固定。
- Fisher 与局部二次近似要求适当数值条件。

判断准则：同时记录实际 KL、clip fraction、收益和 value 误差；clipping 生效不是改善证明。

### 适用边界

- 声称 PPO clipping 严格限制所有状态的 KL 或保证单调提高收益。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [策略梯度与 actor–critic](../../textbook/policy.md)：用旧策略分布下的局部代理目标近似真实收益变化。

- 改变信息或数据协议 · [流式更新与稳定性](../../textbook/streaming.md)：多轮批内优化不同于每步只消费一次新经验。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

改变动作概率还会改变未来状态分布，旧数据只直接描述旧策略。

### 本章的核心思路

限制更新尺度以维持局部比较的可信度，并明确理论界与实际近似之间的差距。

1. [从真实性能差分找到近似位置](trust-region.md#lesson-derive)：用旧占用分布替代新占用分布是关键近似。

2. [用局部几何求受限步长](trust-region.md#lesson-local)：TRPO 借助 Fisher、共轭梯度和回溯近似求解信赖域问题。

3. [用裁剪代理简化优化](trust-region.md#lesson-ppo)：PPO 对有利的比率变化设置分支，但没有把实际约束精确解出来。

结论与条件：理想信赖域理论有优势和分布条件；实际平均 KL 与裁剪目标不自动继承全部保证。

### 相关方法改变了什么

- TRPO / PPO：分别近似求约束更新与优化裁剪代理目标。

- Clip / KL penalty：前者截取目标分支，后者在目标中惩罚偏离，二者都需观测实际分布变化。


<a id="lesson-setting"></a>

## 1 · 旧策略的数据与新策略的状态分布

策略梯度给出局部上升方向，却没有给出安全步长。同一梯度方向走得太远，策略可能进入旧数据很少覆盖的状态。令旧策略为 $\pi_0$，候选策略为 $\pi_\theta$，优势为 $A^{\pi_0}$。本节的理论采用奖励有界、$0<\gamma<1$ 的无限时域折扣 MDP。训练代码则采用有限 episode、$\gamma=1$；它使用同类 surrogate，不直接满足下面折扣定理的所有条件。

$$
d_\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(S_t=s),\qquad r_\theta(s,a)=\frac{\pi_\theta(a\mid s)}{\pi_0(a\mid s)}
$$

$d_\pi$ 是归一化折扣状态分布，所有策略共用同一初始分布，$J(\pi)=\mathbb E_\pi\sum_t\gamma^tR_{t+1}$。概率比只修正给定状态中的动作分布，不会自动把旧状态分布变成新状态分布。

本章使用覆盖候选动作的旧策略。若旧策略对某个动作的概率为零，普通重要性比没有定义；小批量估计也无法凭空得到该动作的优势。

<a id="lesson-derive"></a>

## 2 · 从性能差分到局部 surrogate

$$
J(\pi_\theta)-J(\pi_0)=\frac{1}{1-\gamma}\mathbb E_{s\sim d_{\pi_\theta},a\sim\pi_\theta}[A^{\pi_0}(s,a)]
$$

将 $A=Q-V$ 展开为一步奖励与两个价值项，再沿新策略轨迹求折扣和；中间价值项望远镜相消。

$$
\mathbb E_{\pi_\theta}\sum_{t=0}^{N-1}\gamma^t A^{\pi_0}(S_t,A_t)=\mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{N-1}\gamma^tR_{t+1}+\gamma^N V^{\pi_0}(S_N)-V^{\pi_0}(S_0)\right]
$$

先在有限 N 上展开，才能看见尚未消去的尾值。奖励有界且 γ<1 使尾值趋零；共同初始分布下，最后的初始价值期望等于旧策略回报。再按 d 的定义重写折扣和，才得到上式。这一步不需要候选策略的 critic。

右侧仍需新策略的状态访问分布。可采样的近似是用 $d_{\pi_0}$ 替换它，再用动作概率比。新旧策略相同时，surrogate 的值与一阶导数都对齐，但远离旧策略后不再是精确性能。

$$
\begin{gathered}L_{\pi_0}(\pi_\theta)=J(\pi_0)+\frac{1}{1-\gamma}\mathbb E_{d_{\pi_0},\pi_0}[r_\theta A^{\pi_0}]\\J(\pi_\theta)\ge L_{\pi_0}(\pi_\theta)-\frac{4\gamma\epsilon_A}{(1-\gamma)^2}\alpha_{\rm TV}^{\,2}\end{gathered}
$$

这里 $\epsilon_A=\max_{s,a}|A^{\pi_0}(s,a)|$。策略差异为 $\alpha_{\rm TV}=\max_s D_{\rm TV}(\pi_0(\cdot\mid s),\pi_\theta(\cdot\mid s))$。这是 TRPO 使用的最坏状态差异形式之一。

这个界解释了为什么要限制策略变化。它也显示理论与实现的距离：算法估计有限样本平均 KL，不直接约束所有状态的最大距离；优势由 critic 估计；共轭梯度和回溯也有数值误差。实际 TRPO 不能因此宣称每一步真实回报必然增加，PPO 更没有继承该保证。

<a id="lesson-local"></a>

## 3 · Fisher、共轭梯度与回溯

$$
\begin{gathered}\max_\Delta g^\top\Delta\quad\text{s.t.}\quad\tfrac12\Delta^\top F\Delta\le\delta\\F=\left.\nabla_\theta^2\mathbb E_{s\sim d_{\pi_0}}D_{\rm KL}(\pi_0\Vert\pi_\theta)\right|_{\theta=\theta_0}\end{gathered}
$$

$g$ 是样本 surrogate 在旧参数处的梯度；$F$ 是 KL 的局部曲率。旧策略的概率必须固定，不参与求导。

$$
x=F^{-1}g,\qquad\Delta_* =\sqrt{\frac{2\delta}{g^\top x}}\,x
$$

此式先假定 $F$ 正定且 $g\ne0$。拉格朗日条件给出 $g=\eta F\Delta$，再把二次约束取等号得到缩放。若梯度为零，则不需要更新；奇异的 $F$ 不能直接求逆。

神经网络不显式存储整个 $F$。给定向量 $v$，二次自动微分计算 $Fv=\nabla_\theta[(\nabla_\theta\bar D_{\rm KL})^\top v]$。共轭梯度只需要这个乘法接口，就能近似解线性系统。加入 $\kappa I$ 阻尼有助于处理奇异和数值噪声，但求解的已是阻尼后的方向。

局部二次近似不保证候选点的真实 KL 合格。回溯依次尝试全步、半步、四分之一步，重新计算样本 surrogate 与实际样本 KL；未满足接受条件则恢复旧参数。回溯使用的仍是当前数据，不是对真实环境性能的证明。

<a id="lesson-ppo"></a>

## 4 · PPO 的符号分支与固定量

$$
L^{\rm clip}(\theta)=\mathbb E\left[\min\left(r_\theta\hat A,\operatorname{clip}(r_\theta,1-\varepsilon,1+\varepsilon)\hat A\right)\right]
$$

最大化这个目标；代码通常对其负数做梯度下降。

$$
\ell(r,A)=\begin{cases}A\min(r,1+\varepsilon),&A\ge0,\\A\max(r,1-\varepsilon),&A<0.\end{cases}
$$

正优势动作的过度概率上升、负优势动作的过度概率下降，不再继续得到相同的 surrogate 奖励。

![直接计算ε=.2时正负优势的PPO单样本目标，显示正优势的右侧平台、负优势的左侧平台，与未裁剪rA比较。](https://yingwen.io/crl-figures/learning-classic-ppo-clip.svg)

原创精确函数图：ε=.2，固定A=±1，旧动作概率.25，r从.4到1.6（当前概率.1到.4），61个函数点；采样/训练更新均为0、种子不适用。不是PPO训练曲线；虚线是未裁剪目标。

先看正优势 A=1。r=.6 时，原目标为.6，裁剪项为.8，取小者仍为.6；r=1.4 时，原目标1.4超过裁剪项1.2，所以目标在右侧变平。再看负优势 A=−1：r=.6 时，两个候选是−.6与−.8，取小者为−.8，在左侧变平；r=1.4 时仍取−1.4，惩罚这个有害方向，而不是也在右侧清除梯度。

$$
\frac{\partial\ell(r,1)}{\partial r}=\begin{cases}1,&r<1+\varepsilon,\\0,&r>1+\varepsilon,\end{cases}\qquad \frac{\partial\ell(r,-1)}{\partial r}=\begin{cases}0,&r<1-\varepsilon,\\-1,&r>1-\varepsilon.\end{cases}
$$

这里只列拐点之外的导数。对策略参数求导还要乘 ∇θr；拐点是不可微点，由实现选择相应次梯度。图中的水平段是不再奖励这一单项继续改善，不是给概率设置硬边界。

这也解释数据冻结的作用。一次 rollout 先保存旧动作概率，再以旧 critic 计算优势；批内多轮优化仅让当前动作概率随 θ 改变，不能顺手把分母或优势重算成新值。停止对旧概率、优势和 critic target 求导，隔离的是本次优化中的责任；它不保证优势准确，也不补回新策略的状态分布。下一批 on-policy 数据到来后，才重新定义这一轮的旧策略。

本图按函数公式直接计算，不会因选择随机种子而改变。[数据 JSON](https://yingwen.io/crl-figures/learning-classic-data.json) 与 `scripts/generate-crl-learning-classic.mjs` 保存每个函数点。实际学到的策略与回报应看本章后面的可复现实验，而不是把横轴 r 看作训练步数；两张单项图也不能代替整批共享参数下的更新分析。

clipping 不是把网络参数投影回某个约束集，也不强制所有概率比留在区间内。共享网络会联动改变其他状态和动作，样本不覆盖的地方更没有直接约束。多轮更新时，旧 log-probability、原始优势和 critic target 全部固定；只有当前策略、当前 critic 和优化器状态变化。

配套代码记录 $\widehat{\mathrm{KL}}=\operatorname{mean}(r-1-\log r)$。在旧策略采样、归一化策略与适当覆盖下，其期望对应旧到新 KL。单批估计仍有误差。代码在更新前检查这个量并提前停止，最后一次更新仍可能越过阈值，所以它不是硬约束。

<a id="experiment-deep-ppo"></a>

### 实验：实验 · PPO 的旧数据、多轮更新与裁剪

PPO 的完整更新流程能否在同一小任务上带来可见收益？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步、同样的 6→32 tanh actor/critic 和初始化；actor Adam 0.003、critic Adam 0.01。每 60 步收集一批，γ=1、GAE λ=0.95，优势标准化，clip=0.2，最多 4 次 actor 更新，估计 KL 超过 0.03 时提前停；critic 做 4 次更新。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** ppo.py 保存采样时的旧 log-prob 和旧价值。裁剪目标对正负优势分别起作用；多轮优化不重算旧策略概率。该实验的 VPG 对照每个完整回合只进行一次策略更新。

**测量。** 纵轴是冻结 argmax 行为的原环境回报。应把数据采集步数、重复梯度更新与 rollout 边界分开；图不是 clip 单组件消融。

```bash
python3 implementations/deep/ppo.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-ppo/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 environment_steps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 第 600 步和第 1200 步，PPO 与 VPG 的平均冻结回报均为 0.94，五个 seed 在这些检查点也相同。这里没有实测终点优势；小链可能无法区分两套训练程序。

**结论边界。** PPO 与 VPG 同时在采样分段、GAE、优势标准化和更新次数上不同。不能把这张图称为“裁剪提升”的因果证据。argmax 行为相同不代表两者动作概率相同。

**继续实验。** 保持同一 rollout、同一更新次数与同一优势，仅开关裁剪。记录概率比、KL 和 clip fraction，再观察重新采样后的收益。

[源码](../../implementations/deep/ppo.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-ppo/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-ppo/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-ppo/curves.json)

<a id="course-state-ratio"></a>

## 4.1 · 动作概率比没有校正所有状态的访问频率

$$
\mathbb E_{s\sim d^{\pi_0},\ a\sim\pi_0}
\left[\frac{\pi(a\mid s)}{\pi_0(a\mid s)}A^{\pi_0}(s,a)\right]
=\mathbb E_{s\sim d^{\pi_0},\ a\sim\pi}A^{\pi_0}(s,a)
$$

在动作支持条件下，一步概率比把动作分布换成 π；等式右边的状态分布仍然是旧的 d。性能差分恒等式要求的是新策略的状态访问分布。

两步例子：起点以概率 p 进入支路，否则直接结束。旧策略 p=0.1，新策略 p=0.2。支路中的动作分布完全不变，所以该状态所有动作的概率比都是一；但支路实际出现的概率翻了一倍。只在支路样本上乘当地动作比，无法制造缺少的访问次数。完整轨迹或前缀比可以校正相应分布，但通常承受更大的方差。

TRPO 用限制策略变化来控制 surrogate 与真实改进的差距。PPO clipping 则限制部分样本继续变好的激励，不是完整的状态分布校正。即使批内 KL 很小，未覆盖状态、近似优势以及变化环境仍可能使真实回报下降。长时任务中，小的局部动作变化可以累积成明显不同的访问路径。

更直接的反例不需要未覆盖状态。单状态、两动作，旧概率各半，旧优势为 $(1,-1)$，$\varepsilon=0.2$。候选概率为 $(p,1-p)$。只要 $p\ge0.6$，两个动作的期望 clipped 目标都合计为 0.2；从 p=0.6 到接近一，目标完全平坦，但旧到新 KL 为 $-\tfrac12\log(4p(1-p))$，可任意大。因此目标达到 clipping 平台，并未定义一个信赖域。

若旧策略的优势估计符号已经错误，限制更新幅度只会让错误方向走得少一些，不会把方向变正确。可先在可精确求值的小 MDP 中同时计算真实性能差、未裁剪 surrogate 和 clipped surrogate，再单独加入 critic 误差。这样才能区分策略更新限制与价值估计本身的作用。

与 CRL 的连接：动力学变化、目标变化和状态构造变化，都会在策略 KL 之外引入偏差。评价 PPO 式持续适应时，需要报告变化后累计收益、恢复时间和行动覆盖，而不仅是每轮 clipping fraction。

<a id="lesson-algorithm"></a>

## 5 · 两类算法的精确更新次序

**算法：本教学实现的 actor 与 critic 分开。共享 encoder 的实现必须另外说明共享参数在各阶段如何改变。**

1. 用旧 actor 和 critic 采样；固定旧 log probability。
1. 用旧 value 计算 GAE；固定 critic target；只标准化 actor 优势。
1. TRPO：计算 $g$，用 Hessian-vector product 与 CG 求方向。
  1. 缩放方向，回溯检查实际样本 KL 和 surrogate；失败则恢复参数。
1. PPO：在每轮梯度更新前计算当前 ratio、clipped loss 与样本 KL。
  1. KL 已超过阈值则退出；否则更新 actor。
1. 用固定回归目标训练 critic；重新收集 on-policy 数据。

Fisher-vector product 用二次自动微分求出，并与 categorical Fisher 的解析式对照。

```python
def fisher_vector_product(logits, vector, damping=0.):
    old_prob = logits.detach().softmax(-1)
    old_logp = logits.detach().log_softmax(-1)
    kl = (old_prob*(old_logp-logits.log_softmax(-1))).sum(-1).mean()
    gradient = torch.autograd.grad(kl, logits, create_graph=True)[0]
    product = torch.autograd.grad((gradient*vector).sum(), logits)[0]
    return product+damping*vector
```

<a id="lesson-example"></a>

## 6 · 一维 TRPO 与两个 clipping 例子

单状态两个动作，$\pi_\theta(1)=\sigma(\theta)$，旧 $\theta=0$，优势分别为 $1,-1$。这里对归一化占用下的 $\mathbb E[r_\theta A]$ 求导，故 $g=0.5$，$F=0.25$；保留性能 surrogate 的 $1/(1-\gamma)$ 时，$g$ 也乘此正数，但它在约束全步的归一化中消去。令 KL 半径 $\delta=0.01$，局部全步为 $\Delta=\sqrt{0.08}\approx0.28284$。新动作一概率约为 $0.57024$，实际旧到新 KL 为 $\log\cosh(\Delta/2)\approx0.009967$，因此全步满足这一例的约束。

PPO 取 $\varepsilon=0.2$。若 $A=2,r=1.4$，目标是 $\min(2.8,2.4)=2.4$。若 $A=-2,r=0.6$，目标是 $\min(-1.2,-1.6)=-1.6$。第二例很容易写反：负优势下概率减少过多时，截断取的是更负的一项。

共轭梯度、单状态精确 KL 回溯和 PPO 符号分支。这个 TRPO 数值核不是完整神经网络训练器。

```python
def ppo_term(ratio, advantage, clip=0.2):
    clipped = min(1 + clip, max(1 - clip, ratio))
    return min(ratio * advantage, clipped * advantage)


def conjugate_gradient(matvec, b, iterations=20, tolerance=1e-12):
    x, residual = [0.0] * len(b), list(b)
    direction = list(residual)
    rr = dot(residual, residual)
    for _ in range(iterations):
        if rr <= tolerance * tolerance:
            break
        product = matvec(direction)
        curvature = dot(direction, product)
        if curvature <= 0:
            raise ValueError('positive curvature is required')
        step = rr / curvature
        x = [v + step * d for v, d in zip(x, direction)]
        residual = [r - step * p for r, p in zip(residual, product)]
        new_rr = dot(residual, residual)
        direction = [r + new_rr / rr * d
                     for r, d in zip(residual, direction)]
        rr = new_rr
    return x


def categorical_kl(old, new):
    return sum(p * math.log(p / q) for p, q in zip(old, new) if p > 0)


def trpo_binary(theta, advantage_one, advantage_zero, delta=0.01):
    """Exact scalar Fisher + actual KL/line search for one-state policy."""
    prob = lambda z: 1.0 / (1.0 + math.exp(-z))
    old_p = prob(theta)
    surrogate = lambda z: prob(z) * advantage_one + (1-prob(z)) * advantage_zero
    gradient = old_p * (1-old_p) * (advantage_one-advantage_zero)
    if abs(gradient) < 1e-14:
        return theta, 0.0, 0.0
    fisher = old_p * (1-old_p)
    natural = gradient / fisher
    full_step = math.sqrt(2*delta/(natural*fisher*natural)) * natural
    for power in range(20):
        step = 0.5**power * full_step
        candidate = theta + step
        kl = categorical_kl([old_p, 1-old_p], [prob(candidate), 1-prob(candidate)])
        gain = surrogate(candidate)-surrogate(theta)
        if kl <= delta and gain >= 0.1 * gradient * step:
            return candidate, kl, gain
    return theta, 0.0, 0.0
```

<a id="lesson-code"></a>

## 7 · 可运行 PPO 与实现边界

旧概率和优势来自固定 rollout。critic 使用未标准化的回归目标。

```python
def ppo_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               old_logp, advantages, returns, epochs=4, clip=.2, target_kl=.03):
    # Old log-probabilities, advantages and targets stay fixed for all epochs.
    old_logp, returns = old_logp.detach(), returns.detach()
    advantages = advantages.detach()
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    updates, final_kl = 0, 0.0
    for _ in range(epochs):
        distribution = Categorical(logits=actor(x))
        logp = distribution.log_prob(actions)
        ratio = (logp-old_logp).exp()
        kl = ((ratio-1)-(logp-old_logp)).mean()
        if float(kl.detach()) > target_kl:
            break
        surrogate = torch.minimum(ratio*advantages,
                                  ratio.clamp(1-clip, 1+clip)*advantages)
        loss_actor = -surrogate.mean()
        actor_optimizer.zero_grad()
        loss_actor.backward()
        actor_optimizer.step()
        updates += 1
    for _ in range(epochs):
        prediction = critic(x).squeeze(-1)
        assert prediction.shape == returns.shape
        loss_critic = ((prediction-returns)**2).mean()
        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()
    with torch.no_grad():
        difference = Categorical(logits=actor(x)).log_prob(actions)-old_logp
        final_kl = float((difference.exp()-1-difference).mean())
    return {'actor_updates': updates, 'sample_kl': final_kl,
            'value_loss': float(loss_critic.detach())}
```

将 deep_textbook_train.py 与 deep_textbook_lab.py 放在同一目录。安装 deep_requirements.txt 后执行 python3 examples/deep_textbook_train.py ppo --epochs 16 --seed 0。该命令在内置小 MDP 中真实采样、计算 GAE、做 PPO 更新并独立评估；没有 Gym 或 GPU 依赖。

完整 TRPO 工程入口是 Spinning Up 的 TensorFlow 1 实现；该项目没有对应的官方 PyTorch TRPO 目录。配套 PyTorch 代码只提供 Fisher-vector product，标准库代码提供 CG 与一维回溯。完整网络 TRPO 还需参数向量化、分布接口和整批 line search。

<a id="lesson-branches"></a>

## 8 · 失效条件与持续适应

- 旧策略被覆盖：若每轮内重新保存当前 log_prob 作为 old，概率比总接近一，算法不再控制相对于采样策略的变化。
- critic target 漂移：每轮重新用已更新 critic 生成优势，会把多种变化混在同一目标中。
- 小 batch 的 KL 噪声：阈值过小可能几乎不更新，阈值过大又不能有效限制分布迁移。
- 变化环境：旧策略数据可以在收集完时就不再代表当前动力学。KL 小只表示策略分布接近，不表示环境未变。

PPO 的参数复用和 replay 不是一回事。它复用的是最近采样策略的一批数据，并保存其动作概率。把任意陈旧任务轨迹放入同一循环，不能只保留 clipped loss 就称为正确的 off-policy 算法。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：KL 小是否证明回报提高？答：不证明。真实优势、覆盖与状态分布误差仍有影响。
- 问：把所有 ratio 直接 clip 后再乘优势，等价于 PPO 吗？答：不等价。PPO 是两个目标取最小，某些变坏方向仍必须保留梯度。
- 问：为何使用 CG 而不直接求逆？答：网络参数很多，存储曲率矩阵需要平方级空间；Hessian-vector product 不需要显式矩阵。
- 实验：把一维例子的 δ 增大，比较二次预测 KL 与实际 KL，并观察回溯是否缩步；再将优势都设为零，确认参数保持不变。



<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](../../examples/deep_textbook_lab.py)

```sh
python3 examples/deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Schulman et al. · Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)：性能下界、局部近似与实际 TRPO 的区别。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：clipped surrogate 与多轮小批量更新。

- [Spinning Up · trpo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py)：官方 TensorFlow 1 TRPO，实现 CG、Hessian-vector product 和回溯。

- [Spinning Up · ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：官方 PyTorch PPO 与 KL 提前停止。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](../tabular/dynamic-programming.md) → [策略梯度定理](../approximation/policy-gradient.md) → [TRPO 与 PPO 的近似](trust-region.md) → [持续控制的比较器](../../textbook/control.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [策略梯度与 actor–critic](../../textbook/policy.md)
- [实验设计、统计与算法测试](../../textbook/experiments.md)

对应原始材料：Sutton & Barto §13.2–13.5；TRPO §3–5；PPO §3。本文为原创讲解，原书、论文与上游代码保留各自许可。
