# 最大熵连续控制：SAC 的价值、密度与温度

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

## 本章内容

- 推导 soft Bellman 与 actor 的 KL 投影。
- 计算 tanh 与动作缩放后的概率密度。
- 区分 actor、critic、温度三类梯度与停止梯度位置。

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

控制目标同时评价外部奖励和策略随机性；随机 actor 的密度进入价值与策略更新。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 可微密度 $\pi_\theta(a\mid s)$、温度 $\alpha\ge0$；连续动作密度相对于指定坐标与基准测度定义。

### 需要求解的对象

求最大熵目标下的策略与 soft 价值，而非只给普通奖励策略增加探索噪声。

### 信息与数据权限

使用 replay 转移和当前 actor 采样动作；需要正确的变换后动作密度。

$$
J_\alpha(\pi)=\mathbb E_\pi\!\left[\sum_{t\ge0}\gamma^t\{R_{t+1}-\alpha\log\pi(A_t\mid S_t)\}\right]
$$

熵项是优化目标的一部分。连续微分熵依赖动作坐标，温度与奖励尺度必须一起解释。

### 成立条件与解的含义

- 目标与对数密度可积；有界动作变换包含 Jacobian。
- 自动温度调整还引入目标熵约束及其可行性。

判断准则：分别检查外部收益、熵、温度和密度数值；不能把 entropy bonus 算入环境奖励率后与无熵算法直接比较。

### 适用边界

- 把自动温度当作任意任务都不需要调目标熵的保证。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [最大熵控制](../../textbook/soft-control.md)：本章用神经 critic、连续 actor 和 replay 近似同一个最大熵问题；与相关章共享 soft 目标，而非再次改变目标。

- 改变评价目标 · [持续控制与学习智能体比较](../../textbook/control.md)：相对只评价外部奖励的控制，本章把策略熵纳入优化目标，因而改变 Bellman 方程与策略改善对象。

- 改变评价目标 · [平均奖励与差分价值](../../textbook/average.md)：本章相对平均奖励控制同时采用折扣时间聚合与熵项；若研究平均奖励 soft 控制，还须另定该目标，不能只将折扣设为一。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

策略既决定采样动作，也决定自己的密度惩罚；有界动作变换会改变密度。

### 本章的核心思路

从最大熵目标推导 soft 备份，再用重参数化构造可微 actor 更新。

1. [先改变价值对象](entropy-control.md#lesson-derive)：熵进入后继 soft value，普通 Q target 不能原样使用。

2. [保持采样动作与密度一致](entropy-control.md#lesson-density)：tanh 变换的 Jacobian 是概率公式的一部分。

3. [把随机性要求写成约束](entropy-control.md#lesson-temperature)：温度更新的符号来自对偶方向，不靠经验记忆正负号。

结论与条件：精确 soft 策略改善与深度 replay 实现不同；函数逼近和数据覆盖仍限制实践。

### 相关方法改变了什么

- 固定 / 自适应温度：前者指定奖励与熵权衡，后者试图满足另给的熵水平。

- TD3 / SAC：不只是确定性与随机性区别，优化目标和 critic 定义也不同。


<a id="lesson-setting"></a>

## 1 · 熵是目标的一部分

SAC 优化奖励和策略熵的加权和。$\alpha>0$ 称为温度。它控制奖励与随机性之间的权衡，不是 actor 的学习率。这里动作连续，熵指微分熵；它依赖坐标单位，也可以为负。确定性评估动作与训练时的随机策略需要分别记录。

$$
\begin{gathered}J_\alpha(\pi)=\mathbb E_\pi\sum_{t\ge0}\gamma^t\left[R_{t+1}+\alpha\mathcal H(\pi(\cdot\mid S_t))\right]\\\mathcal H(\pi)=-\mathbb E_{a\sim\pi}\log\pi(a\mid s)\end{gathered}
$$

奖励和熵均按时间累计。实际 SAC 使用 replay 的经验状态分布进行回归和策略改进，该分布不必等于当前策略的折扣占据分布。

更高的单步动作熵不保证发现遥远目标。对多个状态都随机抖动，仍可能无法形成连贯的探索轨迹。最大熵目标、内在奖励和有记忆的探索策略是不同机制。

<a id="lesson-derive"></a>

## 2 · soft Bellman 与策略改进

$$
\begin{gathered}Q^\pi(s,a)=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})\mid s,a]\\V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\alpha\log\pi(a\mid s)]\end{gathered}
$$

此约定的 Q 不含当前已指定动作的熵项，V 才对当前动作分布加入熵；Q 的未来展开包含后续熵。

固定状态与 Q，选择策略最大化 $\int\pi(a)[Q(a)-\alpha\log\pi(a)]\,da$。用拉格朗日乘子满足密度积分为一，对 $\pi(a)$ 求变分导数，得到 $Q(a)-\alpha(1+\log\pi(a))+\eta=0$。归一化后即下面的 Boltzmann 密度。有限配分函数与适当可积性是该推导的条件。

$$
\begin{gathered}\pi^*(a\mid s)=\frac{\exp(Q(s,a)/\alpha)}{Z(s)}\\D_{\rm KL}\!\left(\pi_\theta\middle\Vert\frac{\exp(Q/\alpha)}{Z}\right)=\frac{1}{\alpha}\mathbb E_{\pi_\theta}[\alpha\log\pi_\theta-Q]+\log Z\end{gathered}
$$

Z 与待优化 actor 参数无关，因此最小化这个 KL 等价于最小化 actor loss。参数化 Gaussian 通常不能精确表示 Boltzmann 密度。

理想改进怎样接到长程价值？在有限动作、有界奖励的情形，若每个状态的新策略都使 $\mathbb E_{\pi_{\rm new}}[Q^{\pi_{\rm old}}-\alpha\log\pi_{\rm new}]$ 不小于旧策略的值，则 soft Bellman 备份满足 $T_{\pi_{\rm new}}V^{\pi_{\rm old}}\ge V^{\pi_{\rm old}}$。其单调性使反复备份保持不小于旧值，压缩性使其收敛到 $V^{\pi_{\rm new}}$。因此结论来自逐状态改善加精确评价，不来自一次 replay 平均 loss 的下降；共享 Gaussian 网络的一步更新一般不是这种理想操作。

常用双 Q 版本没有单独的 value network：下一状态的 actor 给出动作与 log-density，target critic 给出价值。这里使用后续 SAC 版本的这一结构；不要把早期带 value network 的实现与当前更新式混在一起。

<a id="course-entropy-units"></a>

## 2.1 · 熵温度有奖励单位，不是通用的稳定性开关

$$
\max_{\pi}\sum_a\pi(a)Q(a)+\alpha\mathcal H(\pi),
\qquad \pi^*(a)=\frac{\exp(Q(a)/\alpha)}{\sum_b\exp(Q(b)/\alpha)}
$$

在一个固定状态，固定 Q，最大化带熵的策略改进目标。拉格朗日乘子处理概率和为一，驻点条件给出 softmax。α 越大，越愿意为动作随机性牺牲 Q。

具体地，约束项写成 $\zeta(\sum_a\pi(a)-1)$。对每个概率求导，得到 $Q(a)-\alpha(\log\pi(a)+1)+\zeta=0$。移项后，$\log\pi(a)=Q(a)/\alpha+\text{常数}$；归一化即得到上式。这里要求 $\alpha>0$，连续动作还需替换为可归一化的密度。

两个动作的 $Q=(0,1)$，$\alpha=1$，高值动作概率为 $e/(1+e)\approx0.7311$。把 Q 的奖励单位放大十倍，却保持 $\alpha=1$，得到约 0.999955。若同时将 $\alpha$ 放大十倍，才恢复原概率。因此奖励归一化若没有相应处理温度，就不仅改变优化数值，也改变熵与回报的权衡。

对于完整 soft Bellman 系统，把奖励与温度共同乘正的常数，会把 soft value 同比例缩放，并保持同一策略改进问题。只缩放奖励并不具有这个不变性。自动温度调节提供另一个约束式目标，但目标熵、估计延迟和优化误差仍需要声明。

动作坐标则改变每步熵的零点。若每一维放大固定倍数，总 log-Jacobian 为 C，熵奖励每个有效步增加 $\alpha C$。在所有策略都持续行动的无限折扣任务中，总增量是常数 $\alpha C/(1-\gamma)$；在真终止后不再累计熵的任务中，总增量为 $\alpha C\,\mathbb E_\pi\sum_{t<T}\gamma^t$。后一个量可以依赖策略。若两种行为分别在一步和两步后终止，新增收益的差就是 $\gamma\alpha C$。动作单位、终止约定与目标熵需要一起定义。

较大的动作熵也不保证访问有用状态。一个 agent 可以在同一小区域内随机晃动，却从未完成通向远端信息的动作序列。深探索需要跨时间一致的行动假设；保持每步分布宽，只控制了局部随机性。

进入 CRL 后，需要分别问：环境是否改变了奖励尺度？原来的熵约束是否仍合适？策略是否还有获取反证的机会？温度更新能够调节随机程度，却不负责发现目标、构造状态、保持表示可塑性或解决不可逆探索。

<a id="lesson-density"></a>

## 3 · 重参数化与 tanh 的 Jacobian

$$
\begin{gathered}\epsilon\sim\mathcal N(0,I)\\u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon\\a=\tanh u\\\log\pi_\theta(a\mid s)=\sum_j\left[\log\mathcal N(u_j;\mu_j,\sigma_j^2)-\log(1-\tanh^2u_j)\right]\end{gathered}
$$

固定基础噪声 $\epsilon$ 后，动作是参数的可微函数。actor 的 $Q$ 项通过动作回传梯度；不能使用不保留路径的普通 sample。

密度变换满足 $p_a(a)=p_u(u)/|\det(\partial a/\partial u)|$。tanh 的导数是 $1-\tanh^2u$，所以 log-density 必须减去 log-Jacobian。各动作维度独立的 Gaussian 输出逐维 log_prob，求和后才得到形状为 batch 的联合动作密度。

$$
\log(1-\tanh^2u)=2\left[\log2-u-\operatorname{softplus}(-2u)\right]
$$

这个等式避免直接计算 $1-\tanh^2u$ 时的浮点消减。饱和区仍可能有很大的密度修正，但不必先算出 $\log(0)$。

若物理动作为 $a^{\rm env}_j=b_j+c_j\tanh u_j$，其中 $c_j\ne0$ 是固定尺度，还需再减去 $C=\sum_j\log|c_j|$。在固定 critic 与温度的一次 actor 更新中，这个密度常数没有直接参数梯度；这并不说明整套学习过程不受影响。熵、soft target 和自动温度都必须采用同一动作坐标。配套 PyTorch 核只实现 $[-1,1]$ 动作；标准库测试另外检查仿射缩放修正。

$$
\mathcal H_{\rm env}=\mathcal H_{\rm norm}+C,\qquad \bar{\mathcal H}_{\rm env}=\bar{\mathcal H}_{\rm norm}+C
$$

同时平移目标熵才能保持同一个熵约束。若固定温度并在真终止后停止累计熵，新增的每步常数还会按存活时间累计；当策略影响 episode 长度时，它可能改变策略偏好。

<a id="lesson-temperature"></a>

## 4 · 熵约束与温度更新的符号

若要求平均熵不低于目标 $\bar{\mathcal H}$，其约束为 $\mathbb E[-\log\pi]\ge\bar{\mathcal H}$。将约束加入奖励优化，可对非负乘子 $\alpha$ 做对偶下降。固定当前策略样本时，温度损失为 $L_\alpha=-\alpha\,\operatorname{sg}(\log\pi+\bar{\mathcal H})$。熵过低意味着括号为正，下降会增加温度。

$$
\begin{gathered}\alpha=e^\beta\\L_\beta=-e^\beta\operatorname{sg}(\log\pi+\bar{\mathcal H})\\\nabla_\beta L_\beta=-\alpha\,\operatorname{sg}(\log\pi+\bar{\mathcal H})\end{gathered}
$$

$\beta$ 保证温度为正。策略样本在温度 loss 中固定，actor 更新时则固定温度。

一些工程采用代理损失 $-\beta\,\operatorname{sg}(\log\pi+\bar{\mathcal H})$。其梯度没有乘 $\alpha$，因此不是上式的精确链式导数。正温度下零点相同，但更新尺度不同。本配套代码实现精确参数化；对照其他仓库时需要检查实际损失，不能只看变量名 log_alpha。

本实现的平均取 replay 状态和当前策略动作；一个共享温度调节这个平均，不对每个状态分别施加硬熵下界。目标还必须可行：有界动作空间上的最大微分熵受其体积限制，策略类也可能进一步限制可达熵。温度持续增加可以暴露约束不可达，不能据此认定探索已经改善。

数值例：$\alpha=0.2,\log\pi=2,\bar{\mathcal H}=-1$。当前样本熵为 $-2$，低于目标 $-1$。于是梯度为 $-0.2$，梯度下降增大 $\beta$ 和 $\alpha$。连续密度可大于一，所以 log-density 为正与概率论并不冲突。

<a id="lesson-algorithm"></a>

## 5 · critic、actor 与温度的更新次序

$$
y=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right],\quad a'\sim\pi_\theta(\cdot\mid s')
$$

整个 y 停止梯度；SAC 使用当前 actor 生成下一动作，而非 TD3 式 target actor。

$$
\begin{gathered}L_{Q_i}=\mathbb E(Q_{\phi_i}(s,a)-\operatorname{sg}(y))^2\\L_\pi=\mathbb E_{\mathcal D,\epsilon}\left[\alpha\log\pi_\theta(a_\theta\mid s)-\min_iQ_{\phi_i}(s,a_\theta)\right]\end{gathered}
$$

actor loss 中 critic 参数固定，动作路径保持可微；温度作为常数。

固定 Q 时，上式是 replay 状态上的 KL 投影目标。精确 soft 策略迭代还要求正确评价和逐状态的相应改善；有限网络共享参数、近似双 critic 与有限数据不满足这些理想操作。因而这一步的 pathwise 梯度可以正确计算实际 loss，却不必是原始起点回报的无偏梯度，也不保证每步真实改善。

**算法：这是配套核选定的次序。温度使用 actor 更新前算出的那批样本；重新采样也是一种实现选择，但必须明确。**

1. 采样 replay batch，读取真正终止标志。
1. 固定温度；无梯度计算下一动作、log-density 与 soft target。
1. 用固定 target 更新两个 critic。
1. 冻结 critic 参数；重参数化采样当前动作。
1. 更新 actor 的 entropy-minus-Q loss；恢复 critic 可训练标志。
1. 使用刚才样本的 detached log-density 更新 log temperature。
1. 仅对两个 target critic 做 Polyak 更新。

<a id="experiment-deep-sac"></a>

### 实验：实验 · 连续 SAC 的梯度路径与真实控制结果

把 tanh 随机策略、双 critic 和固定温度接成完整训练循环后，结果是什么？

**环境与可用信息。** BoundedLQ：观测为位置 x 和剩余时间比例；动作 u∈[−1,1]。位置按 0.92x+0.3u 更新并截到 [−3,3]；奖励为 $-(x^2+0.05u^2)$。初态均匀取自 [−1,1]，40 步是真实有限时域终止。

**设置。** 1200 个真实步，种子默认初始化 32 单元 tanh 隐层；SAC Gaussian actor、两个 critic，温度固定 0.1，不学习温度。actor Adam 0.001，critic Adam 0.002，γ=0.99，replay 4000、批量 32；第 32 步起每步更新，前 64 步均匀探索，目标软更新新参数占比 0.02。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** sac.py 的 critic target 使用当前随机策略和目标双 Q；actor 更新冻结 Q 参数但保留 Q 对动作的导数。_common.py 的 GaussianActor 在 tanh 后校正 log-prob。对照 DDPG 使用确定性 actor、单 critic 和标准差 0.15 的动作噪声。

**测量。** 每 60 步用固定评价种子 991 的 12 个回合评价。SAC 执行 tanh(mean)，不是随机动作；纵轴是未折扣的外部奖励总和，不含熵项。

```bash
python3 implementations/deep/sac.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-sac/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 environment_steps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** SAC 的平均评价回报从初始化 −21.609 变为第 600 步 −21.157，再到第 1200 步 −0.690；DDPG 对应为 −5.967、−4.474、−2.353。初期 SAC 明显更差，后期均值更高；DDPG 末端 seed 标准差约 2.350，不能只报两个终点数字。

**结论边界。** actor 架构和初始行为分布不同，即使 seed 相同也不是完全同参数对照。这是两个完整方法的短任务比较，不隔离熵、双 Q 或随机策略的单项作用；也不覆盖自动温度和单次生命持续任务。

**继续实验。** 在同一网络、replay 和数据预算下比较固定温度的多个取值。分别报告随机行为外部收益、确定性评价和含熵目标，解释三者为何可能不同。

[源码](../../implementations/deep/sac.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-sac/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-sac/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-sac/curves.json)

<a id="lesson-example"></a>

## 6 · soft target 与密度手算

取奖励一、$\gamma=0.9$、两个 target Q 的最小值二、$\log\pi=-0.5$、$\alpha=0.2$。soft value 为 $2-0.2(-0.5)=2.1$，target 为 $1+0.9\times2.1=2.89$。相比同一数值下 TD3 的 2.8，多出的不是环境奖励，而是目标内的熵项。

一维标准 Normal 在 $u=0$ 的 log-density 为 $-\frac12\log(2\pi)\approx-0.91894$。这里 tanh 导数为一，所以 squash 修正为零。若再映射到 $[-2,2]$，需要减去 $\log2$，得到约 $-1.61209$。这解释了为何温度目标不能脱离动作尺度。

Normal.rsample 保留动作路径；稳定 Jacobian 与动作维度求和组成完整的 log-density。

```python
class SquashedGaussian(nn.Module):
    def __init__(self, observation_dim=3, action_dim=2):
        super().__init__()
        self.net = mlp(observation_dim, 2*action_dim)

    def forward(self, observation):
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = u.tanh()
        log_jacobian = 2*(math.log(2)-u-nn.functional.softplus(-2*u))
        log_prob = (normal.log_prob(u)-log_jacobian).sum(dim=-1)
        return action, log_prob
```

<a id="lesson-code"></a>

## 7 · 可执行的 SAC 更新核

双 critic、重参数化 actor 与精确 log-temperature 对偶下降。

```python
def sac_update(actor, q1, q2, target_q1, target_q2, log_alpha,
               actor_opt, critic_opt, alpha_opt, batch, target_entropy=-2., gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_action, next_logp = actor(xp)
        soft_value = torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))-alpha*next_logp
        y = reward+gamma*(1-terminal)*soft_value
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([q1, q2], True)
    sampled_action, logp = actor(x)
    assert logp.shape == reward.shape
    loss_actor = (alpha*logp-torch.minimum(q1(x, sampled_action), q2(x, sampled_action))).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([q1, q2], False)
    # Exact log-alpha parameterization of the dual loss, not its common proxy.
    loss_alpha = -(log_alpha.exp()*(logp.detach()+target_entropy)).mean()
    alpha_opt.zero_grad(); loss_alpha.backward(); alpha_opt.step()
    update_targets([q1, q2], [target_q1, target_q2], tau)
    return {'critic_loss': float(loss_q.detach()), 'actor_loss': float(loss_actor.detach()),
            'alpha': float(log_alpha.exp().detach())}
```

执行 python3 examples/deep_textbook_lab.py test 检查 tanh 密度、动作缩放、soft target 与温度有限差分；执行 python3 examples/deep_textbook_train.py test 检查动作与 log-probability shape、可微采样路径和实际 SAC 更新。此处只提供更新核，没有完整连续环境采样、replay 训练和 benchmark。

Spinning Up 的 SAC 教学实现使用固定 alpha，适合对照核心损失和停止梯度；自动温度见 SAC Algorithms and Applications。作者早期 sac 仓库与后续 softlearning 仓库所处算法版本不同，不能认为所有源码都应含有相同网络。

<a id="lesson-branches"></a>

## 8 · 失败条件：密度、尺度与数据覆盖

- 对 Gaussian 动作直接 clip，却仍使用原 Gaussian 密度：边界出现概率质量，原密度公式失效。
- 多维 log_prob 未求和：temperature loss 与 actor loss 的维度和尺度都可能错误。
- reward scale 改变而 alpha 不变：奖励与熵的相对权重随之改变，已经不是同一优化问题。
- 只检查两个 critic 的 loss：actor 仍可能访问 critic 外推区域，低训练误差不代表可靠控制。

在持续任务中，温度调节回答的是随机性目标是否满足，不回答表征是否足够、旧任务是否遗忘或参数是否失去可塑性。它是一种特定学习参数的自适应机制，不能代表所有元学习问题。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：SAC 的温度等于 actor 学习率吗？答：不是。温度出现在目标里，学习率控制优化该目标时的参数步长。
- 问：为什么 actor 更新不能 detach sampled_action？答：Q 项需要通过动作对 actor 求导；detach 会删除这一项。
- 问：连续动作的目标熵为负是否错误？答：不错误，微分熵可为负并依赖动作单位。
- 实验：将动作尺度从一改为二，确认 log-density 减少 log2；保持目标熵不变时观察温度梯度如何变化，再讨论怎样平移目标以保持同等约束。



<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](../../examples/deep_textbook_lab.py)

```sh
python3 examples/deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Haarnoja et al. · Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)：ICML 2018 最大熵 actor–critic 原论文。

- [Haarnoja et al. · Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)：后续 SAC 结构与自动温度的约束推导。

- [Spinning Up · sac.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)：官方 PyTorch 教学实现，温度 alpha 固定。

- [Haarnoja · sac](https://github.com/haarnoja/sac)：作者早期实现；包含的算法结构与后续双 Q、无独立 V 版本需分别对照。

- [RAIL · softlearning](https://github.com/rail-berkeley/softlearning)：作者团队后续连续控制框架；使用其配置时需记录版本与算法设置。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 优化一段折扣回报，还是长期单位时间收益？

回报定义决定策略排序。折扣奖励、有限时域和平均奖励是不同目标；把折扣取大，只是在某些条件下接近相应极限。

函数逼近与深度方法：持续任务的相对价值需要奖励率和定标条件。训练中更新策略时，奖励率也在变化。动作时长不同，还要区分每次决策与单位物理时间的收益。

持续学习中的研究问题：长期收益率忽略有限的启动损失；单生命期不能忽略。怎样同时报告生命期收益、适应成本和后期表现，并让预测、控制与模型使用一致的时间单位？

[平均奖励控制基础](../approximation/average-control.md) → [熵如何改变目标](entropy-control.md) → [平均奖励的预测、控制与规划](../../textbook/average.md) → [完整学习器的评价](../../textbook/control.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [最大熵控制](../../textbook/soft-control.md)
- [策略梯度与 actor–critic](../../textbook/policy.md)
- [探索与经验选择](../../textbook/exploration.md)
- [元学习与学习规则的适应](../../textbook/meta.md)

对应原始材料：SAC §4；SAC Algorithms and Applications §5；Spinning Up SAC key equations。本文为原创讲解，原书、论文与上游代码保留各自许可。
