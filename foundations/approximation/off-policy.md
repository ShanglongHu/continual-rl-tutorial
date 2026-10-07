# 离策略函数逼近：覆盖、发散与稳定更新

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

## 本章内容

- 区分动作重要性修正与状态加权带来的稳定性。
- 推导 MSPBE 及 GTD2、TDC 的辅助权重更新。
- 理解 emphatic weighting 的目的、时间索引和线性理论条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 两种策略与覆盖

行为策略 $b$ 生成经验，目标策略 $\pi$ 指定待预测的未来。只要 $\pi$ 对某个状态—动作给正概率，$b$ 也必须给正概率，才可能使用有限的重要性比。

$$
\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)
$$

### 投影固定点

固定特征下，TD 求解期望更新为零的方程。这里的状态权重来自行为分布，而下一步目标来自目标策略。

<a id="problem-definition"></a>

## 本章的问题定义

行为策略生成数据，目标策略定义预测对象。共享参数和自举可能使朴素校正 TD 不稳定。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 给定目标策略 $\pi$，$v_\pi(s)=\mathbb E_\pi[\sum_{k\ge0}\gamma^kR_{t+k+1}\mid S_t=s]$。
- 误差权重 $d(s)\ge0$ 且 $\sum_sd(s)=1$；权重指定在哪些状态上评价估计。
- 行为策略 $b$、状态权重矩阵 $D$、特征矩阵 $X$，投影 $\Pi_D$。

### 需要求解的对象

稳定求解指定的离策略预测问题，而非改变目标策略来适应数据。

### 信息与数据权限

目标动作由行为覆盖，且动作概率比可得；状态分布通常仍是行为分布。

$$
\operatorname{MSPBE}(w)=\|Xw-\Pi_DT_\pi Xw\|_D^2
$$

这是线性函数空间中的投影 Bellman 误差。它与真实价值均方误差、逐样本 TD 平方都不同。强调方法又改变投影的权重结构。

### 成立条件与解的含义

- 目标对行为绝对连续；所需矩阵与采样矩满足相应定理条件。
- 覆盖是必要信息条件，不足以保证半梯度迭代稳定。

判断准则：既检查概率比与时间索引，也检查固定点、矩阵谱和参数是否有界。

### 适用边界

- 从离策略预测收敛推导深度最优控制收敛。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [价值预测与资格迹](../../textbook/value.md)：改变数据策略不应改变所声明的目标价值。

- 组合不同学习问题 · [通用价值函数与预测知识](../../textbook/gvf.md)：多个目标策略的预测可共享行为流，但每个预测仍有自己的稳定性条件。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

重要性采样能校正动作分布，却不自动使自举更新成为下降方向。

### 本章的核心思路

先用反例定位不稳定，再选择显式误差梯度或改变状态权重。

1. [区分覆盖与稳定性](off-policy.md#offpolicy-counterexample)：即使每个目标动作都有数据，更新矩阵仍可能放大误差。

2. [用辅助变量估计目标梯度](off-policy.md#offpolicy-gtd)：GTD2/TDC 的第二组参数并非额外价值任务，而是校正更新方向。

3. [通过跟随迹重加权状态](off-policy.md#offpolicy-emphasis)：ETD 改变有效更新分布，因此与仅校正动作概率不是同一机制。

结论与条件：结论首先针对线性预测和明确步长、矩阵条件；高方差和控制耦合仍需单独处理。

### 相关方法改变了什么

- GTD2 / TDC：辅助估计相近，主参数更新的梯度与校正形式不同。

- 梯度 TD / ETD：前者显式处理投影目标梯度，后者改变有效状态权重。


<a id="lesson-setting"></a>

## 1 · 一个经验流支持多个预测

机器人只实际执行一个动作，却可能同时预测“继续前进会碰撞吗”和“转向后能否充电”。这些预测的目标策略与正在执行的策略不同。经验回放也会使用旧策略的数据。离策略学习因而不是一种特定控制算法，而是一种数据来源与目标行为分离的设定。

$$
\begin{aligned}&\mathbb E_b[\rho_t f(S_t,A_t,S_{t+1})\mid S_t=s]\\ &\qquad=\mathbb E_\pi[f(s,A_t,S_{t+1})\mid S_t=s].\end{aligned}
$$

重要性比修正给定当前状态时的动作分布。它没有把当前状态的访问权重 $d_b(s)$ 自动变成 $d_\pi(s)$，更没有改变特征共享。

覆盖保证需要的数据原则上能够被观察。概率极小时，权重可能很大，估计方差也很大。将比值截断或归一化可以改变方差，但通常也改变所估计的量。除了支持条件，还要检查状态分布、特征和更新算子的组合是否稳定。

可把离策略问题拆成两个不同的选择：未来按照谁来预测；现在优先在哪些状态预测准确。π 决定第一项，D 决定第二项。它们可以故意不同。例如行为会频繁经过某个路口，学习器希望在这里准确预测“如果向左走”的结果，不必假装自己已经长期生活在向左策略诱导的平稳分布里。

$$
\mathbb E_b\!\left[\frac{d(S_t)}{d_b(S_t)}\rho_t f_t\right]=\sum_s d(s)\mathbb E_\pi[f_t\mid S_t=s].
$$

要求目标评价权重 $d$ 被 $d_b$ 覆盖。若希望改成特定 $d$，可同时校正状态比和动作比；但 $d_b$ 及目标状态比通常未知，不能把这条恒等式直接当作免费可用算法。

Emphatic TD 不是简单估计 dπ/db。它通过 interest 与 follow-on trace 构造另一个有利于稳定性的强调分布。Gradient TD 则保持指定投影目标、改造更新方向。两者处理同一个普通 TD 可能失稳的现象，却改变了不同的对象；比较时应报告最终价值误差及其加权分布，不仅报告参数是否停止发散。

<a id="rlss-excursion-objective"></a>

## 评价假想的一生，还是从实际处境出发的预测

动作重要性比回答“接下来若按目标策略行动，会发生什么”。它没有回答“在哪些处境上的预测应该准确”。后一个问题在近似表示下不可省略：共享参数通常不能同时消除全部状态的误差。先固定环境和两种策略，并假定相应平稳分布存在。

$$
J_{\mathrm{life}}(w)=\sum_s d_\pi(s)[v_\pi(s)-\widehat v(s,w)]^2,\qquad J_{\mathrm{exc}}(w)=\sum_s d_b(s)i(s)[v_\pi(s)-\widehat v(s,w)]^2.
$$

前者按一直执行目标策略时的状态分布评价；后者按实际行为遇到的处境及指定 interest 评价。它们使用同一个目标价值函数，却分配不同的近似资源。乘一个公共正的归一化常数不改变最小点。

假设实际行为在 A、B 各停留一半，而目标策略从任何状态都转到 B。目标策略的长期分布只保留 B。如果只按这个分布优化，共享 critic 可以牺牲 A 的准确性。但“现在身处 A，如果改用目标策略会怎样”仍是行为智能体可能需要的问题。Excursion 视角保留这个问题；它不要求智能体真的切换策略，也不把预测范围限制为下一步。

从回合起点累乘过去动作的重要性比，可以把实际轨迹的历史分布改成目标轨迹的历史分布。若还要把未来回报改成目标策略，就另需未来的校正。长历史的乘积可能具有很高方差。稳态下直接估计密度比是另一条路线，但需要其自身的估计条件。以上都不同于仅在当前 TD 误差前乘一个动作比。

ETD 的 follow-on 分布也不是简单地把行为分布还原成目标的稳态分布。它从行为遇到且有 interest 的状态出发，沿目标策略追踪自举依赖。它建立的是一个强调加权的投影方程；不能把该方程的解直接称为上式任一 MSVE 的全局最小点。

在持续 GVF 中，行为可能不断变化，却同时保留许多目标策略预测。这时必须分别说明预测语义、评价处境、更新权重，以及用什么时间窗评价。固定策略的稳态分析是参照模型；对不断变化的行为，不能默默把一个不存在的长期平稳分布当作已知常量。

<a id="offpolicy-counterexample"></a>

## 2 · 覆盖充分却发散的二状态反例

有 A、B 两状态和“去 A”“去 B”两个动作。行为在每个状态都以 0.1 概率去 B、0.9 概率去 A，所以平稳状态权重是 (0.9,0.1)。目标总去 B；它的动作被行为覆盖。令奖励全为 0，$\gamma=0.9$，一维特征 $x_A=1,x_B=2$。

$$
\begin{aligned}A&=\mathbb E_b[\rho x(x-\gamma x')]\\ &=0.9(1)(1-1.8)+0.1(2)(2-1.8)\\ &=-0.68.\end{aligned}
$$

目标的真实价值为零。但目标始终指向大特征状态，加上行为状态权重后，平均 TD 更新的反馈符号反转。

$$
\begin{aligned}h(w)&=\mathbb E_{S\sim d_b,\,A\sim b}[\rho\delta_w x]=0-Aw=0.68w,\\ w^+&=w+\alpha h(w)=(1+0.68\alpha)w.\end{aligned}
$$

先冻结 w，再对指定平稳状态动作分布求平均。这个确定性均场递推对任何正步长都有增长方向。实际在线 $w_t$ 与经验流相关，不能把 $\alpha h(w_t)$ 无条件写成真实的 $\mathbb E[\Delta w_t\mid w_t]$；这里也不声称每个采样步都增大权重。减小步长不改变这个均场固定点的不稳定性。

图中目标箭头总指向大特征状态，而行为大多停留在小特征状态。把这两种比例代入更新，原本要纠正预测的反馈变成了放大器。下面两条对照只改变这个具体模型的数据分布或自举项。

![两个共享权重状态、目标策略箭头与行为状态质量，以及原反例和两项单机制对照的精确均场递推。](https://yingwen.io/crl-figures/concept-classic-td-stability.svg)

原创精确递推图，$w_0=1,\alpha=0.01$，共 300 次固定数据律的均场更新，非在线随机训练曲线。原反例乘子为 1.0068；改用目标策略平稳分布时为 0.996；保留行为状态权重但令 $\gamma=0$ 时为 0.987。三条曲线共用坐标。这里的稳定对照是本例的计算结果，一般收敛仍需其余条件。

这展示了“致命三元组”：自举、离策略和函数逼近同时存在时可能不稳定。它不是说每一个包含三者的算法都会发散，也不是说去掉任一项就无需其他条件。神经网络、目标网络与回放会改变动力学，需要另外检查。

<a id="experiment-baird_expected_td"></a>

### 实验：实验 · 去掉噪声与非线性，TD 仍能发散

已知真值可表示、输入已归一化且每步使用精确期望时，离策略半梯度为什么仍会越学越错？

**环境与可用信息。** Baird 星形七状态、八维固定线性特征，每个特征向量单位长度。目标策略总转移到下状态，行为状态分布均匀。奖励全零，γ=0.99，真值全零。

**设置。** 五种子各运行 1200 个精确七状态 sweep，不是 1200 个环境步。步长 0.05，初始参数接近全 1，其中下状态专有坐标为 10，其他坐标加入 ±0.01 的种子扰动。对照用同一已知模型求 Bellman 残差真梯度。

**检验的机制。** 所有状态项先用同一旧参数计算再同时更新。没有采样噪声、神经网络、回放或优化器矩。差异来自更新方向及其优化目标；残差梯度对照不是把半梯度 TD 的同一个固定点简单稳定化。

**测量。** 图中纵轴为 log10(1+价值 RMSE)，只压缩显示尺度。原始 rmse 与 parameter_norm 都保留，训练没有裁剪。不要把纵轴 2.64 误读为原始误差只有 2.64。

```bash
python3 implementations/nonlinear_diagnostics/baird_expected_td.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/baird_expected_td/curves.svg)

训练种子 0、1、2、3、4；每种方法 1200 expected_sweeps。阴影为 ±1 个样本标准差，不是置信区间。

**结果分析。** 半梯度 TD 的平均图值由约 0.529 增至 2.641；五种子原始 RMSE 末尾约为 436–438。残差梯度末尾图值约 0.271，未发生同样增长。该反例直接说明不稳定并非只能归因于深网或随机噪声。

**结论边界。** 精确模型 sweep 不是严格流式算法；特征冗余，参数解不唯一。有限预算中残差梯度仍有非零价值误差，不能把下降曲线写成已经达到真值。

**继续实验。** 对期望 TD 更新矩阵检查特征值，再将步长减半并延长横轴，区分“增长较慢”与“根本稳定”。保持原始 RMSE 和对数图同时可见。

[源码](../../implementations/nonlinear_diagnostics/baird_expected_td.py) · [逐种子记录](https://yingwen.io/crl-code/results/baird_expected_td/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/baird_expected_td/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/baird_expected_td/curves.json)

<a id="rlss-baird-representability"></a>

## Baird 反例：能表示真值，为什么仍然发散

近似误差与更新不稳定是不同困难。标准 Baird 反例甚至可以表示全部七个状态的任意价值，却仍能让普通离策略 TD 发散。因而不能把它简单解释成“参数太少，装不下答案”。

六个上方状态记为 1 到 6，下方为 7。solid 动作从任何状态转到 7；dashed 动作均匀转到六个上方状态。目标总选 solid，行为以 1/7 概率选 solid、6/7 选 dashed。因此行为长期均匀访问七状态，目标长期只访问状态 7。奖励全为 0，γ=0.99；真实目标价值也全为 0。

$$
\widehat v(i)=2w_i+w_8\quad(i=1,\ldots,6),\qquad \widehat v(7)=w_7+2w_8.
$$

采用课程的未归一化特征约定。令 w₈=0，并分别取 wᵢ=vᵢ/2、w₇=v₇，即可表示任意七状态价值。矩阵有七个独立行和一个参数零空间；参数冗余不等于价值表达不足。

为看清增长方向，取六个上方权重相同。对称性使精确期望更新保持这个条件。记共同的上方预测为 u，下方预测为 v。修正动作比后，每个当前状态仍以 1/7 权重参与。把八个参数更新投影回两种预测，得到一个二维闭合递推。

$$
\Delta w_i=\frac{2\alpha}{7}(\gamma v-u)\ (i\leq6),\quad
\Delta w_7=\frac{\alpha}{7}(\gamma-1)v,\quad
\Delta w_8=\frac{\alpha}{7}\bigl[6(\gamma v-u)+2(\gamma-1)v\bigr].
$$

对每个当前状态，先用 TD 误差乘该状态的特征，再按七分之一求和。前六个参数各只接收一个上方状态的更新；第八个共享参数接收全部七状态的更新。

$$
\Delta u=2\Delta w_i+\Delta w_8,\qquad
\Delta v=\Delta w_7+2\Delta w_8.
$$

代入上一式即可逐项得到下式的四个矩阵元素。发散通路正是这种共享：为了提高一个状态的预测，同时改变了作为它自举目标的下方预测。

$$
\begin{pmatrix}u^+\\v^+\end{pmatrix}=\left[I+\alpha B\right]\begin{pmatrix}u\\v\end{pmatrix},\qquad B=\frac17\begin{pmatrix}-10&12\gamma-2\\-12&17\gamma-5\end{pmatrix}.
$$

上方误差为 γv−u，下方误差为 (γ−1)v。一个上方参数只影响其本状态；共享的第八个参数同时影响全部上方状态和下方状态。汇总这些效应即得到 B。

$$
\operatorname{tr}(B)=\frac{17\gamma-15}{7}>0,\qquad \det(B)=\frac{26(1-\gamma)}{49}>0\quad(\gamma=0.99).
$$

此时两个实特征值约为 0.23925 与 0.02218，均为正。相应期望更新乘子为 1+αλ>1；减小正步长只会减慢增长。零初始化会停在真值，但不能据此认为其他初始化也稳定。

共享参数把后继预测也一同抬高，行为状态权重又没有按目标轨迹提供相应的校正。即使遍历全部状态、使用精确期望，也保留这种反馈，所以它不是采样噪声造成的。GTD、ETD 或不同表示改变的是更新几何或权重，不能只凭更多采样期待符号翻转。

诊断时分别记录价值误差、参数范数与 MSPBE。参数可沿零空间变化而不改变预测；反过来，一个很小的投影目标在病态问题中也未必对应很小的真实价值误差。这个反例是固定线性预测的压力测试，不是所有深度控制失效的统一解释。

<a id="lesson-derive"></a>

## 3 · 三种误差与 MSPBE 的推导

$$
\begin{aligned}\mathrm{MSBE}&=\|T_\pi\Phi w-\Phi w\|_D^2,\\ \mathrm{MSPBE}&=\|\Pi_D T_\pi\Phi w-\Phi w\|_D^2.\end{aligned}
$$

MSBE 衡量 Bellman 残差，MSPBE 先将残差投影到可表示空间。这里 $D=\operatorname{diag}(d_b)$。二者均不同于对真实价值的误差。

样本均方 TD 误差又是第三个量。先保持与 MSBE 相同的状态权重 $d_b$，但在给定状态后按目标策略 π 采动作和环境后果；否则原始行为样本的平方误差回答的是不同策略的问题。冻结 w 时，“先平方再求期望”与“条件期望后平方”的区别为下面的条件方差。

$$
\mathbb E_{d_b,\pi}[\delta_w^2]=\mathrm{MSBE}(w)+\mathbb E_{S\sim d_b}[\operatorname{Var}_\pi(\delta_w\mid S)],\qquad \mathbb E_{d_b,b}[\rho\delta_w^2]=\mathbb E_{d_b,\pi}[\delta_w^2].
$$

动作重要性比只乘一次。$\mathbb E_b[(\rho\delta)^2]$ 一般不是这个目标策略样本损失。要求动作支持与有限二阶矩；动作比不改变当前状态权重。

直接对同一个目标策略后果样本的平方残差求梯度，通常不能得到 MSBE 的无偏梯度；条件独立的双采样是一个经典障碍。重要性比修正动作分布，并没有消除下面的同样本协方差。

$$
\begin{aligned}\bar\delta_w(s)&=\mathbb E[\delta_w\mid S=s],\\\nabla_w\tfrac12\bar\delta_w(s)^2&=\mathbb E[\delta_w\mid s]\,\mathbb E[\nabla_w\delta_w\mid s],\\\mathbb E[\delta_w\nabla_w\delta_w\mid s]&=\mathbb E[\delta_w\mid s]\,\mathbb E[\nabla_w\delta_w\mid s]+\operatorname{Cov}(\delta_w,\nabla_w\delta_w\mid s).\end{aligned}
$$

先假定可对同一真实状态重复采样目标策略后果。MSBE 梯度含两个条件期望的乘积。把它们都用同一次后继样本替代，会多出向量协方差项；两个条件独立后继样本可以消去这个偏差。

固定当前特征 x=1、奖励 R=1、γ=.5、w=1。下一特征 x′ 等概率为0或2。两种后果的 δ 分别为0、1，对 w 的导数分别为−1、0。因此真实局部残差平方梯度是 E[δ]E[∂δ]=.5×(−.5)=−.25，但同样本乘积的期望是0。算法可能以为已经没有梯度，实际目标仍有下降方向。这个单状态条件算例只检查估计器，不声称构造了某个完整控制环境。

双采样困难不是说 Bellman 方程错误，也不是任何模型下都无法估计 MSBE。可重置到同一真实状态的模拟器或准确生成模型能提供额外条件。但单条不可重置轨迹通常没有同一时刻的第二个独立后果。若两个不同真实状态又映射到相同观测特征，数据可能根本无法辨认应按哪个真实状态分组；这是表示与可辨识性问题，比有限样本方差更深。

$$
\begin{aligned}C&=\mathbb E_b[xx^\top],\\ A&=\mathbb E_b[\rho x(x-\gamma x')^\top],\\ b_v&=\mathbb E_b[\rho Rx],\\ J(w)&=\tfrac12(b_v-Aw)^\top C^{-1}(b_v-Aw).\end{aligned}
$$

用 $b_v$ 表示奖励向量，避免与行为策略 $b$ 混淆。将投影矩阵代入 MSPBE 并约去 $C$ 后，得到这个二次目标。要求特征在行为分布下独立，使 $C$ 可逆。

$$
\begin{aligned}-\nabla J(w)&=A^\top C^{-1}(b_v-Aw),\\ h&\approx C^{-1}(b_v-Aw).\end{aligned}
$$

辅助权重 $h$ 学习一组预条件化的期望 TD 更新。它不是第二个价值函数，而是为了用单个经验流跟踪梯度中的期望。

<a id="rlss-td-nonconservative"></a>

## 半梯度为什么不一定是任何标量损失的梯度

“没有对 bootstrap 目标求导”不仅是省掉一个计算步骤。得到的更新向量场可能根本不是某个固定标量函数的梯度。可以不用复杂 MDP，就检查这个局部数学事实。

固定重复使用一个转移样本，x=(1,0)、x′=(0.3,1)、奖励 0、γ=0.9。这里故意只研究这个样本的更新场，不把它冒充完整的 on-policy 轨迹。

$$
F(w)=\delta x=\begin{pmatrix}-0.73w_1+0.9w_2\\0\end{pmatrix},\qquad \frac{\partial F_1}{\partial w_2}=0.9\ne0=\frac{\partial F_2}{\partial w_1}.
$$

若 F 是某个二次可微标量函数的梯度，它的 Jacobian 必须对称。这里交叉偏导不同，所以不存在这样的全局势函数。沿单位正方形逆时针积分 F·dw，还会得到 −0.9 而非零。

这不表示 TD 没有意义。固定点、算子压缩与随机逼近稳定性，并不要求每一步都下降同一个损失。反之，把样本 δ² 当作可打印的日志量，也不意味着 TD 在对它做梯度下降。若真正把 δ² 对两端预测都求导，得到的是残差梯度路线，仍需面对目标选择、双采样及下一节的可辨识性问题。

<a id="rlss-bellman-identifiability"></a>

## A-split 与 Split-A：哪些目标无法由同一经验流辨认

双采样是估计难题；隐藏状态混叠还能造成更根本的可辨识性问题。若两个世界给出完全相同的可观测经验分布，却要求不同的最优参数，任何只读取该经验的算法都无法保证选对。下面给出完整的两种世界。

世界甲从 A 开始。以相同概率立即终止并得到 0，或先以奖励 0 到 B，再以奖励 1 终止。世界乙的起点以相同概率为 A₁、A₂：A₁ 确定地以奖励 0 到 B，A₂ 确定地以奖励 0 终止；B 仍以奖励 1 终止。学习器在 A、A₁、A₂ 都只收到同一个特征 a，在 B 收到 b。两个世界产生相同的完整观测回合。

令共同的 A 类预测为 u，B 的预测为 v，γ=1。按每回合的期望状态访问次数加权。以下目标未除以期望回合长度 3/2；这个公共正因子不改变最小点。

$$
J_{\mathrm{BE}}^{\rm A\text{-}split}=(\tfrac12v-u)^2+\tfrac12(1-v)^2,\qquad (u^*,v^*)=(\tfrac12,1).
$$

在世界甲，对真正状态 A 的后继先取条件期望，然后平方。u=v/2 消除 A 的期望残差，v=1 消除 B 的残差。

$$
J_{\mathrm{BE}}^{\rm Split\text{-}A}=\tfrac12(v-u)^2+\tfrac12u^2+\tfrac12(1-v)^2,\qquad (u^*,v^*)=(\tfrac13,\tfrac23).
$$

在世界乙，应先分别对隐藏的 A₁ 与 A₂ 计算残差，再求加权平方和。令 2u−v=0、2v−u−1=0 得到不同的最小点。

可观测的一步样本 TD 平方误差在两个世界都等于第二式。把相同特征当成同一真实状态，再按特征合并样本，只能得到一个由特征条件定义的目标；它不必等于每个潜在 MDP 的状态级 MSBE。知道真实状态身份、获得相应重置接口或额外模型信息，会改变这个不可能性论证的前提。

$$
\mathbb E[(G-\widehat v(S))^2]=\mathbb E[(v_\pi(S)-\widehat v(S))^2]+\mathbb E[\operatorname{Var}(G\mid S)].
$$

固定策略时，最后一项不依赖预测参数。因此平方回报误差与平方价值误差具有同一最小点，即使后者的数值本身无法仅由混叠观测确定。可辨认目标数值、可辨认最小点和能否高效学到，是三个问题。

在这两个世界中，A 类完整回报都是等概率的 0 或 1，MC 的最小点均为 (1/2,1)。世界乙在这个点仍有无法消除的真实状态价值误差。MSPBE 则由已观察到的特征转移矩决定；在矩条件和可逆性成立时，它可通过这些矩估计。能估计这个目标，不代表它等于 MSVE，也不保证任意 TD 更新稳定。

这一反例直接通向 agent state：加入记忆是否真的区分了与未来有关的历史？如果在动作前没有任何线索区分 A₁、A₂，仅换成 RNN 也不能凭空恢复隐藏标签。先写清学习器能见到什么，再决定损失与可检验的正确性标准。

<a id="offpolicy-gtd"></a>

## 4 · GTD2 与 TDC 的两组更新

$$
\begin{aligned}h^+&=h+\beta[\rho\delta-x^\top h]x,\\ w^+_{\mathrm{GTD2}}&=w+\alpha\rho(x-\gamma x')(x^\top h).\end{aligned}
$$

GTD2 直接采样主参数的梯度方向。两个式子都使用旧 $h$ 和旧 $w$；不能用刚更新的辅助参数替代同一步公式中的 $h$。

$$
w^+_{\mathrm{TDC}}=w+\alpha\rho\big[\delta x-\gamma x'(x^\top h)\big].
$$

TDC 保留普通 TD 项，再加修正项。辅助参数仍按上一式更新。在 $h=C^{-1}(b_v-Aw)$ 时，TDC 的期望方向与 MSPBE 的负梯度一致；有限时刻的样本方向并不要求相同。

尤其注意：辅助更新的协方差项 $-(x^\top h)x$ 不整体乘 $\rho$。这组公式选择行为状态下的 $C$。当某个动作使 $\rho=0$ 时，主参数不更新，但辅助参数仍可以沿协方差项变化。代码为这个容易写错的情况设置了测试。

**算法：Gradient TD 的单步时序**

1. 初始化主权重 $w$、辅助权重 $h$，固定目标与行为策略
1. 读取转移及采样时的行为概率，检查支持并计算 $\rho$
1. 用旧参数计算 $\delta$ 和 $x^\top h$
1. 按 GTD2 或 TDC 计算完整主参数增量
1. 按辅助方程计算增量，最后同时写入两个新参数

经典理论针对固定线性特征、固定策略、适当矩阵非奇异性和矩条件，并要求与分析匹配的采样和递减步长。GTD2 的联合系统与 TDC 的两时间尺度分析需要区分；TDC 常令辅助参数比主参数更快，步长比趋于零。本页固定步长的小例子只检验公式和动力学，不是这些随机收敛定理的复现。

<a id="offpolicy-emphasis"></a>

## 5 · Emphatic TD：改变状态更新权重

另一条路线改变各状态的更新强调程度。非负 interest $i(s)$ 指定哪些预测重要；follow-on trace 追踪重要性沿目标策略的后继传播。设 $0\le\lambda_t\le1$，初始化 $F_{-1}=0,e_{-1}=0$。这种加权改变了投影几何，不是在原来 TD 更新外附加一个新的奖励。

$$
\begin{aligned}F_t&=i_t+\gamma_t\rho_{t-1}F_{t-1},\\ M_t&=\lambda_t i_t+(1-\lambda_t)F_t,\\ e_t&=\rho_t(\gamma_t\lambda_t e_{t-1}+M_t x_t),\\ w_{t+1}&=w_t+\alpha_t\delta_t e_t.\end{aligned}
$$

$\gamma_t$ 描述进入当前状态的延续，$\delta_t$ 的 bootstrap 使用 $\gamma_{t+1}$。$F_t$ 用上一动作比值，$e_t$ 用当前比值。interest 不是额外奖励。

![兴趣只在A注入，follow-on沿实际采样路径用上一比率递推；每个时刻再乘当前比率，得到不同的特征更新系数。](https://yingwen.io/crl-figures/concept-credit3-emphatic-flow.svg)

图示同一组递推的具体时间顺序。设 interest 为 (1,0,0)，首次进入之后 γ=0.5，采样动作比率为 (2,4,1)。F 依次为 (1,1,2)，λ=0 下当前特征系数为 (2,4,2)。状态下方的 $c_{t+1}$ 在本列动作之后收到；它不参与 F 的递推。原创数值算例，采用固定单热特征。

目标策略沿 $A\to B\to C\to A$ 前进，行为在 A、B 分别以 0.5、0.25 的概率前进，否则等待，在 C 必回 A。图中是一段恰好连续前进的实际经验，累计信号 c 取到达 C 的指示。虽然只对 A 直接设置了 interest，A 的预测依赖 B，B 又依赖 C；将 B、C 的学习一律关闭会丢掉这些自举依赖。图中的 F 是这段样本历史产生的权重，不是把行为状态频率直接变成目标策略稳态频率的密度比。

有限状态 ETD 理论要求固定目标满足 $(I-P_\pi\Gamma)^{-1}$ 存在，行为链不可约且覆盖目标动作，强调为正的状态具有足够独立特征。奖励噪声方差有界；原论文还给出特定递减步长条件，例如适当的 $\alpha_t=a/(b+t)$，其中 $a,b>0$。表示学习、变化策略和常数步长追踪，不直接落在这一定理内。

强调提高稳定性不等于低方差。多个重要性比沿时间传播可能产生大幅 trace。记录 trace 范数和重尾更新很重要；若裁剪 trace，则应说明已改变算法。没有必要为了避免普通 TD 发散而隐藏修复方法自身的方差代价。

<a id="rlss-emphasis-geometry"></a>

## Interest 是要求，emphasis 是自举依赖所产生的权重

Interest 表示我们在哪些状态关心预测。一步自举使这些预测依赖后继状态；因此，即使某状态 interest 为零，也可能需要在它上面认真学习。ETD 的 follow-on trace 正是把这种依赖纳入更新权重，而不是把用户偏好直接当成采样权重。

$$
f(s)=d_b(s)\,\mathbb E[F_t\mid S_t=s],\qquad f=D_b i+\gamma P_\pi^\top f,\qquad f=(I-\gamma P_\pi^\top)^{-1}D_bi.
$$

这里固定策略、固定 γ<1，并取平稳极限；f 是未归一化的状态权重。由 F 的递推，对前一状态和动作求和得到中间式。ETD(0) 的期望更新由这个 f 加权，而非简单使用 dπ 或 db·i。

两状态例子：行为长期各访问一半，目标从任何状态都转到 B。只关心 A，所以 i=(1,0)，取 γ=0.9。方程给出 f_A=0.5、f_B=0.9(f_A+f_B)=4.5。B 自己没有 interest，却承接 A 的预测依赖。归一化后权重为 (0.1,0.9)，不同于目标策略的长期分布 (0,1)。

这不是承诺最小化 interest 加权的真实价值误差。ETD 修改了自举方程的投影权重，目的是在规定的固定线性设定下建立稳定更新。若想比较“在哪些状态更准确”，应另行报告指定 interest 下的价值误差及实际强调分布。

$$
F_t=1+\gamma\rho_{t-1}F_{t-1},\qquad \mathbb E[\rho]=1,\qquad \mathbb E[F]=\frac1{1-\gamma}.
$$

再取每步独立选动作的简单行为，ρ 以概率 1/7 为 7，其余为 0，interest 恒为 1。新比值独立于已有 F，故这个均值递推成立。γ=0.99 时长期均值为 100。

$$
\mathbb E[(\gamma\rho)^2]=7\gamma^2=6.8607>1\quad(\gamma=0.99).
$$

若假设平稳 F 存在有限二阶矩，将递推平方取期望会得到 E[F²]=1+2γE[F]+7γ²E[F²]，左右无法由非负有限数满足。因此有限均值不意味着有限方差。

这解释了为什么用精确期望强调矩阵得到平滑下降曲线，不能冒充逐样本 ETD 的实际表现。后者偶尔出现很长的重要性比乘积。记录中位数、较高分位数、最大 trace 和累计更新贡献，往往比只报告平均 F 更有用。裁剪或截断 trace 是合理的研究选择，但改变了算法，需另外解释偏差。

把这一差别做成可运行实验。环境只有一个状态，两个动作都回到自己。目标始终选择动作 a；行为每步独立地以 p=1/7 选择 a。因此重要性比为 7 或 0。设 γ=0.99、interest=1，从 F₀=1 开始。这里不训练 critic，也不比较控制收益；只隔离强调权重的采样问题。

$$
\mathbb E[F_t]=\sum_{j=0}^{t}\gamma^j=\frac{1-\gamma^{t+1}}{1-\gamma},\qquad \mathbb E[F_t^2]=1+2\gamma\mathbb E[F_{t-1}]+\frac{\gamma^2}{p}\mathbb E[F_{t-1}^2].
$$

当前动作独立于已有 F，且 E[ρ]=1、E[ρ²]=1/p。每个有限 t 的二阶矩仍有限，却随 t 急剧增长；上面的无限二阶矩结论针对稳态。不能把两种时限混为一谈。

完整实验保留所有路径；每条路径先采样上一动作比，再递推到下一时刻。精确矩递推与采样循环分别实现。

```python
def followon_step(previous_followon, previous_ratio, gamma, interest=1.0):
    """Compute incoming F_t. The ratio belongs to action t-1, NOT action t."""
    return interest + gamma * previous_ratio * previous_followon


def finite_trace_moments(time, gamma, probability):
    """Exact moments from F_0=1 and iid previous rho in {0, 1/p}.

    This is deterministic moment propagation, not a sampled trace or a critic.
    The caller uses small time values for second moments to avoid float overflow.
    """
    mean, second = 1.0, 1.0
    for _ in range(time):
        second = 1.0 + 2.0 * gamma * mean + gamma * gamma / probability * second
        mean = 1.0 + gamma * mean
    return mean, second


def sample_trace_batch(seed, paths, horizon, gamma, probability, checkpoints):
    """Independent one-state trajectories, with both actions self-looping.

    pi(a)=1 and b(a)=p. At every action: rho=1/p with probability p, else 0.
    Every path is retained. No clipping, outlier removal, or stopping rule.
    Samples at DIFFERENT times within a path are not independent replicates.
    """
    rng = random.Random(seed)
    values = [1.0] * paths  # F_0=1; no sampled previous action yet
    rows = []
    selected = set(checkpoints)
    target_action_count = 0
    for time in range(horizon + 1):
        if time in selected:
            rows.append({"time": time, **describe(values)})
        if time == horizon:
            break
        for path in range(paths):
            chose_target = rng.random() < probability
            target_action_count += int(chose_target)
            previous_ratio = 1.0 / probability if chose_target else 0.0
            values[path] = followon_step(values[path], previous_ratio, gamma)
    return {"seed": seed, "checkpoints": rows, "target_action_count": target_action_count,
            "actions": paths * horizon, "final_values": values}
```

![精确有限时刻的 follow-on 均值，与四个预先指定独立批次的样本均值比较。](https://yingwen.io/crl-code/diagnostics/rlss-prediction/followon-ensemble.svg)

γ=0.99、p=1/7。每批 2048 条独立路径，512 个动作，种子为 0、1、2、3；共 4194304 次动作采样。纵轴为对数。没有裁剪、弃样或种子筛选，也没有置信阴影。每条曲线是在同一时刻对该批全部路径求平均，不是把相关时间点当作独立重复。

在 t=512，精确有限时刻均值为 99.4234。四个批次的均值约为 4.4815、3.5792、5.2666、6.9840。合并 8192 条路径后的均值为 5.0779，中位数为 1，95% 分位数为 7.93，99% 分位数为 55.9549。这些分位数描述路径分布，不是总体均值的置信区间。这个样本均值在固定 t 仍是无偏估计量；这次实现值远低于期望，不等于存在系统偏差。这不否定大数定律；它说明有限的实际预算未必足以充分覆盖尾部。

为什么多条路径的均值仍可能很小？设 K 为当前状态之前连续选择 a 的次数。稳态下，绝大多数时刻刚被 ρ=0 重置；极少数时刻却有很长的连乘积。我们可以直接计算这些罕见事件对期望的贡献，而不要求有限模拟碰巧采到它们。

$$
\Pr(K\ge k)=p^k,\qquad F=\sum_{j=0}^{K}(\gamma/p)^j,\qquad \mathbb E[F\mathbf1\{K\ge k\}]=\sum_{j=0}^{k-1}p^{k-j}\gamma^j+\frac{\gamma^k}{1-\gamma}.
$$

对 F 的非负项逐项求期望。j<k 的项只要求 K≥k；j≥k 的项要求 K≥j。由此得到两部分。该式是稳态解析结果，与上一图的有限初始化采样分开。

先计算尾部事件的概率，再计算它贡献的期望；二者不是同一个百分比。

```python
def stationary_tail(streak, gamma, probability):
    """Exact tail probability and share of the stationary follow-on mean.

    K counts consecutive target actions immediately BEFORE the current state.
    P(K >= k)=p^k and F=sum_{j=0}^K (gamma/p)^j.
    Swapping these nonnegative sums gives
      E[F * 1{K>=k}] = sum_{j<k} p^(k-j)*gamma^j + gamma^k/(1-gamma).
    This avoids division by gamma/p-1, including the gamma=p case.
    """
    validate_trace(gamma, probability)
    if not isinstance(streak, int) or streak < 0:
        raise ValueError("streak must be a nonnegative integer")
    prefix = math.fsum(probability ** (streak - j) * gamma ** j for j in range(streak))
    contribution = prefix + gamma ** streak / (1.0 - gamma)
    return {"streak": streak, "probability": probability ** streak,
            "mean_contribution": contribution,
            "fraction_of_mean": contribution * (1.0 - gamma)}
```

![连续目标动作的尾部事件概率，与它贡献的 follow-on 总体均值占比。](https://yingwen.io/crl-code/diagnostics/rlss-prediction/followon-tail-mass.svg)

单状态模型的稳态解析值，不是模拟估计。横轴为至少连续 k 次目标动作；纵轴为对数占比。两条曲线的分母不同：一条是全部时刻的概率质量，另一条是总期望 E[F]=100。没有随机误差条。

例如 K≥4 只占约 0.04165% 的时刻，却贡献约 96.22% 的稳态均值。K≥12 的概率约为 7.22×10⁻¹¹，仍贡献约 88.79% 的均值。这些事件彼此包含，贡献不能相加。这里的长尾是由完整分布推导出来的，不是看到少数尖峰后猜测的。

研究诊断：把 γ 从 0.99 改成 0.2，保持 p 不变。此时 γ²/p 从 6.8607 降到 0.28，有限稳态二阶矩不再被这个递推排除。再分别改变每批路径数和轨迹长度，检查它们改变的是采样误差还是预测定义。注意：改变 γ 也改变了该预测的延续规则，不能把曲线更平滑解释为免费解决了原来的问题。

源码和全部参数、种子、数值在[独立 Python 实验](../../examples/rlss_prediction_diagnostics.py)与[结果 JSON](https://yingwen.io/crl-code/diagnostics/rlss-prediction/results.json)。只依赖 Python 标准库。单独运行 test 会穷举短动作序列，对照矩递推；run 会生成两项诊断的完整绘图数据。

下载脚本后运行；默认完成 follow-on 分布和共享特征两个诊断

```bash
python3 examples/rlss_prediction_diagnostics.py test
python3 examples/rlss_prediction_diagnostics.py run --output results.json
```

这不是 ETD 参数发散的实验。它没有学习价值函数，更没有使用神经表示。它只说明：平均更新矩阵的性质与实际 trace 分布是两层证据。控制 trace 方差、保持目标语义和证明学习稳定，需要继续分别研究。

进入持续 GVF 与神经表示后，目标策略、interest 和特征都可能变化。原来的稳定性结论不会自动跟随这些变化。可先固定表示、增加并行问题数量，检查少数重尾预测是否污染共享参数，再加入表示学习。一次短任务没有发散，与一个长期自主知识系统足够可靠，不是同一个检验。

<a id="rlss-emphasis-time-structure"></a>

## 把 interest、λ 和时间位置放进同一条计算链

为什么 on-policy 也会需要 emphasis？因为选择性预测与自举会一起改变资源分配。取一条回合轨迹，进入终点前 γ=1，行为就是目标策略，因此 ρ=1。每个回合清空 follow-on trace。先只看前四个非终止时刻。

$$
F_t=i_t+\gamma_t\rho_{t-1}F_{t-1},\qquad M_t=\lambda_t i_t+(1-\lambda_t)F_t,\qquad F_{-1}=0.
$$

γₜ 是到达当前状态的延续因子。ρₜ₋₁ 修正到达当前状态之前的动作。Mₜ 决定本时刻的强调，不是本时刻实际参数更新的全部系数。

| Interest：t=0,1,2,3 | λ | M₀,M₁,M₂,M₃ | 含义 |
| --- | --- | --- | --- |
| 1,0,0,0 | 全为 1 | 1,0,0,0 | 完整回报只需给起点直接分配兴趣 |
| 1,1,1,1 | 全为 1 | 1,1,1,1 | 各时刻按各自完整回报学习 |
| 1,0,0,0 | 全为 0 | 1,1,1,1 | 起点的一步自举依赖传播到后继 |
| 1,1,1,1 | 全为 0 | 1,2,3,4 | 后继累积承接更多早期预测的依赖 |

这不是“越晚的状态天然越重要”。最后一行来自全程 interest、未终止以及一步自举三个具体选择。若 γ=0.5、只关心起点且 λ=0，权重变成 1、0.5、0.25、0.125。若 λ 只在第三个时刻降到 0，起点 interest 会在那个自举位置产生额外强调。

$$
M_t=i_t+\sum_{k=0}^{t-1}M_k\rho_k\!\left(\prod_{j=k+1}^{t-1}\gamma_j\lambda_j\rho_j\right)\gamma_t(1-\lambda_t).
$$

这给出递推的来源。k 时刻的预测沿中间各步继续展开，再在 t 时刻自举，于是它把自身强调的一部分传到 t。空乘积为 1。将这些依赖合并可得到上面的两个标量递推，无需存储整个历史。对应 Mahmood 等原论文式 (10)–(11)。

因此 λ 在这里不只是一个统一的“长短回报旋钮”。位置相关的 λ 指定在哪里使用另一个预测。若前四步 interest 都为 1，且 λ=(1,1,0,1)，则 M=(1,1,3,1)。F 仍累计为 (1,2,3,4)，但只有发生自举的位置接收这部分强调。

再看一个可以手算的预测问题：A 确定地到 B，B 确定地到终点，两步奖励均为 1。γ=1。两个状态使用同一个常数预测 θ，所以真值 2 和 1 不能同时表示。分析时冻结 θ，汇总一回合的更新方向；这对应小步长平均动力学，而不是常数步长下每步参数都变化的精确轨迹。

$$
\delta_A=1,\qquad\delta_B=1-\theta.\qquad U_{\mathrm{TD}}=\delta_A+\delta_B=2-\theta,\qquad U_{\mathrm{ETD}}=\delta_A+2\delta_B=3-2\theta.
$$

普通 TD(0) 的平均固定点为 2。对两个状态都设 interest=1，ETD(0) 的强调为 (1,2)，平均固定点为 1.5。均匀 MSVE 分别为 0.5 与 0.25。后者恰好是本例的最优常数预测；一般情形没有这个最优性保证。

若按 A、B 顺序在线更新并保持常数 α，回合末固定点分别为 2−α 与 1.5−α，且回合内有振荡。普通 TD 需 0<α<2、此 ETD 需 0<α<1，才能使对应回合映射收缩。因此必须标明曲线采集在回合内哪个时刻，不能用有限步长结果机械验证平均固定点。

$$
\theta_{n+1}^{\mathrm{TD}}=(1-\alpha)\theta_n+\alpha(2-\alpha),\qquad \theta_{n+1}^{\mathrm{ETD}}=(1-2\alpha)\theta_n+\alpha(3-2\alpha).
$$

θₙ 是第 n 个回合末的参数。A 更新先把 θₙ 加上 α；随后 B 使用已经变化的参数，分别加上 α(1−θₙ−α) 或 2α(1−θₙ−α)。解这两个仿射递推，便得到回合末固定点及有限回合的精确轨迹。

同一确定性环境中的逐步 TD/ETD，与冻结参数整回合方向。后者使用已知模型，不能算作相同交互预算下的另一个在线算法。

```python
def shared_episode(theta, alpha, emphatic=False):
    """Online A -> B -> terminal; r=1 at both transitions, x(A)=x(B)=1.

    At A: gamma_current=0 resets F to 1; gamma_next=1 bootstraps B.
    At B: gamma_current=1 makes F=2; gamma_next=0 stops bootstrapping.
    The two steps use their own CURRENT theta. This timing matters.
    """
    records = []
    followon = 0.0
    for state, gamma_current, gamma_next in (("A", 0.0, 1.0), ("B", 1.0, 0.0)):
        followon = 1.0 + gamma_current * followon  # rho_previous=1, interest=1
        emphasis = followon if emphatic else 1.0
        old_theta = theta
        delta = 1.0 + gamma_next * old_theta - old_theta
        theta = old_theta + alpha * emphasis * delta  # ETD(0), rho_current=1
        records.append({"state": state, "before": old_theta, "after": theta,
                        "delta": delta, "emphasis": emphasis})
    return theta, records


def frozen_episode(theta, alpha, emphatic=False):
    """Known-model diagnostic, NOT an online two-transition implementation.

    Both residuals are evaluated at the SAME old theta, then summed.
    There is no division by two: alpha multiplies the whole episode direction.
    """
    delta_a = 1.0 + theta - theta
    delta_b = 1.0 - theta
    emphasis_b = 2.0 if emphatic else 1.0
    return theta + alpha * (delta_a + emphasis_b * delta_b)


def shared_closed_form(episode, alpha, emphatic=False, online=True, initial=0.0):
    """Independent analytic trajectory for the episode-END parameter."""
    equilibrium = 1.5 if emphatic else 2.0
    if online:
        equilibrium -= alpha
    contraction = 1.0 - (2.0 if emphatic else 1.0) * alpha
    return equilibrium + (initial - equilibrium) * contraction ** episode
```

![从零初始化开始的在线 TD、在线 ETD，以及分别冻结参数后计算整回合方向的参数曲线。](https://yingwen.io/crl-code/diagnostics/rlss-prediction/shared-feature-fixed-points.svg)

A→B→终止，两步奖励均为 1，两个状态共用 θ。α=0.1，初值 0，200 回合；每个在线方法获得 400 次转移。每点在回合末采集。固定参数对照只检查平均方向，不是逐步学习。实验完全确定，无多种子平均或置信阴影。

| 更新过程 | 200 回合末 θ | 冻结该 θ 后的均匀 MSVE | 要核对的区别 |
| --- | --- | --- | --- |
| TD 逐步在线 | 约 1.9 | 约 0.41 | 回合末点为 2−α |
| ETD 逐步在线 | 约 1.4 | 约 0.26 | 回合末点为 1.5−α |
| TD 冻结参数整回合方向 | 约 2.0 | 约 0.50 | 零方向不等于有限步长在线周期 |
| ETD 冻结参数整回合方向 | 约 1.5 | 约 0.25 | 本例恰好达到最优常数预测 |

![最后四个回合内逐转移记录的参数，显示 A 更新后与 B 更新后的周期。](https://yingwen.io/crl-code/diagnostics/rlss-prediction/shared-feature-cycle.svg)

奇数横坐标为 A 更新后，偶数为 B 更新后。TD 在约 2.0 与 1.9 之间变化；ETD 在约 1.5 与 1.4 之间变化。振荡由常数步长和更新顺序产生，不是环境噪声，也不是置信带。

这个图不建立 ETD 在任意任务中优于 TD 的排序。它核对的是三个对象：平均方向的零点、有限步长的在线周期，以及给定表示下的误差。把 α 减半，回合末偏移和周期幅度也减半；但同样回合预算下，早期适应会更慢。再把两个状态改为独立参数，检查哪些差别来自表示共享，哪些来自更新时序。

完整[脚本](../../examples/rlss_prediction_diagnostics.py)和[实际结果](https://yingwen.io/crl-code/diagnostics/rlss-prediction/results.json)同时包含 follow-on 分布和共享特征两个诊断。独立闭式轨迹对全部 200 个回合、四种更新的核对最大绝对误差低于 10⁻¹⁴。这个误差验证实现与推导一致，不是算法性能的统计显著性。

$$
e_t=\rho_t\bigl(\gamma_t\lambda_t e_{t-1}+M_t\phi_t\bigr),\quad \delta_t=R_{t+1}+\gamma_{t+1}\theta_t^\top\phi_{t+1}-\theta_t^\top\phi_t,\quad \theta_{t+1}=\theta_t+\alpha_t\delta_t e_t.
$$

完整 ETD(λ) 还需要资格迹。当前 ρₜ 修正当前动作；它和 F 里的上一动作比不可互换。λ=0 才化为单步的 Mₜρₜδₜφₜ。这里是普通 accumulating ETD，不是 true-online ETD，也不是 GTD 的校正公式。

例如 i=(1,0,0,0,0,0,0)、γ=1、λ=0、ρ=(1,2,2,2,0.25,1,0)，得到 M=(1,1,2,4,8,2,2)。最后一个 M 不为零，因为它由过去到达该状态的动作决定；但当前 ρ=0，使当前 e 和参数更新为零。把比值索引错一格，会得到另一个算法。

在固定 γ<1、固定 λ、恒定 interest 的 on-policy 持续轨迹中，F 最终趋近 i/(1−γ)，M 也趋近一个公共常数。此时它主要整体缩放步长；初始化阶段仍有暂态。回合终止、时变 interest 或位置相关 λ 会打破这个简化。不能把这一个特例推广为“on-policy 总不需要 emphasis”。

$$
A_T=\sum_{t<T}M_t\rho_t\phi_t(\phi_t-\gamma_{t+1}\phi_{t+1})^\top,\qquad b_T=\sum_{t<T}M_t\rho_t R_{t+1}\phi_t,\qquad A_T\theta=b_T.
$$

若保存矩估计再求解，可得到 emphatic LSTD(0) 的对应方程。有限样本 A 可能奇异；正则化改变所解问题。稠密矩阵需 O(d²) 存储，直接求解通常需 O(d³) 计算。它不保留 ETD 的每步 O(d) 成本。

这些例子先把目标权重、自举位置和资格迹分开。进入共享神经表示后，再问：新预测的 interest 从何产生？改变 λ 是否改变了依赖图？重尾强调会不会主导共享层？动态特征会不会使旧资格迹失去原梯度含义？线性固定特征的收敛定理不能替这些问题作答。

<a id="lesson-example"></a>

## 6 · 手算辅助变量与强调的时间索引

令 $w=1,h=0,x=1,x'=2,R=0,\gamma=0.9,\rho=2,\alpha=0.1,\beta=0.2$。误差 $\delta=0.8$。GTD2 第一步主权重仍为 1，因为旧辅助权重为零；辅助权重变为 0.32。TDC 第一步则得到主权重 1.16。

对于 ETD，取旧 $F=2$、上一比值 3、进入当前状态的折扣 0.5、interest 为 1。得到 $F_t=1+0.5\times3\times2=4$。若 $\lambda=0,\rho_t=4,x_t=1$，则新 trace 为 16。把当前比值错放进 follow-on，会得到不同答案。

二状态反例中 $C=1.3,b_v=0$，因此 $J(w)=0.68^2w^2/(2\times1.3)$。这是以零为最小点的凸二次函数。GTD2 的期望递推可以下降，而普通 TD 的期望递推增长；代码把它们在同一概率模型中比较。

<a id="lesson-code"></a>

## 7 · 可运行的稳定性与时序检查

GTD2、TDC、ETD 更新核与概率加权反例

```python
def gradient_td(weights, auxiliary, x, reward, xp, gamma, rho, alpha, beta, method):
    if method not in ("GTD2", "TDC") or rho < 0:
        raise ValueError("method must be GTD2/TDC and rho nonnegative")
    delta = reward + gamma * dot(weights, xp) - dot(weights, x)
    hx = dot(auxiliary, x)
    if method == "GTD2":
        direction = [rho * (v - gamma * vp) * hx for v, vp in zip(x, xp)]
    else:
        direction = [rho * (delta * v - gamma * vp * hx) for v, vp in zip(x, xp)]
    next_weights = [w + alpha * d for w, d in zip(weights, direction)]
    # rho multiplies delta, NOT the covariance term hx.
    next_auxiliary = [h + beta * (rho * delta - hx) * v for h, v in zip(auxiliary, x)]
    return next_weights, next_auxiliary


def emphatic_td(weights, trace, followon, previous_rho, x, reward, xp,
                gamma_current, gamma_next, rho, interest, lam, alpha):
    followon = interest + gamma_current * previous_rho * followon
    emphasis = lam * interest + (1 - lam) * followon
    trace = [rho * (gamma_current * lam * e + emphasis * v) for e, v in zip(trace, x)]
    delta = reward + gamma_next * dot(weights, xp) - dot(weights, x)
    weights = [w + alpha * delta * e for w, e in zip(weights, trace)]
    return weights, trace, followon, rho


def off_policy_demo():
    # b(go-to-B)=.1 in each state, pi(go-to-B)=1. d_b=(.9,.1).
    # Features x(A)=1, x(B)=2. All rewards zero. A=-.68, C=1.3.
    samples = [(0.81, [1.0], [1.0], 0.0), (0.09, [1.0], [2.0], 10.0),
               (0.09, [2.0], [1.0], 0.0), (0.01, [2.0], [2.0], 10.0)]
    w, h, td = [1.0], [0.0], 1.0
    for _ in range(2000):
        increments_w, increments_h = 0.0, 0.0
        for probability, x, xp, rho in samples:
            wn, hn = gradient_td(w, h, x, 0, xp, 0.9, rho, 0.01, 0.05, "GTD2")
            increments_w += probability * (wn[0] - w[0])
            increments_h += probability * (hn[0] - h[0])
        w, h = [w[0] + increments_w], [h[0] + increments_h]
        td += 0.01 * 0.68 * td
    return {"A": -0.68, "C": 1.3, "expected_TD_after_2000": td,
            "expected_GTD2_after_2000": w[0],
            "sampling": "exact probability-weighted updates, not a stochastic benchmark"}
```

反例使用全部四种状态—动作组合的精确概率加权，不把单次随机曲线当作期望。运行 2000 次期望更新后，普通 TD 权重约为 769864，GTD2 约为 0.000551。数值展示一个反例的差异；它不说明 GTD2 在所有任务上都更快或更好。

运行离策略反例与更新测试

```sh
python3 examples/approximation_textbook_lab.py off-policy
python3 examples/approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 8 · 从预测理论到 CRL 的边界

多个 GVF 可以共享一个行为流，但每个预测仍须定义自己的目标策略、信号、折扣和 interest。共享表示会耦合更新；线性固定特征下某个预测器稳定，不代表联合神经表示一定稳定。

环境或表示变化时，应分别监测覆盖、价值误差、重要性比、辅助参数范数和 trace 尾部。发散可能来自动态算子、数值尺度、非平稳目标或缺失信息。更换优化器未必解决投影几何，扩大 replay 也未必改善目标动作的覆盖。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 为什么重要性比已经正确，TD 仍发散？答案：它只修正条件动作分布；行为状态加权与共享特征仍可能形成不稳定的投影更新。
- GTD2 第一步主权重不变，是否说明程序失效？答案：若辅助权重初始化为零，这是公式要求；后续辅助信息才驱动主更新。
- $\lambda=1$ 时，ETD 的 $M_t$ 是什么？答案：$i_t$，但 trace 仍含重要性比与历史递推。
- MSPBE 小是否保证控制策略好？答案：不保证。它是给定表示和权重下的预测准则，还要检验表示误差与决策收益。



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](../../examples/approximation_textbook_lab.py)

```sh
python3 examples/approximation_textbook_lab.py off-policy
python3 examples/approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton et al. · Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation](https://icml.cc/2009/papers/546.pdf)：GTD2、TDC 与投影误差推导的原始论文。本文代码额外显式写出目标与行为动作的重要性比。

- [Mahmood et al. · Emphatic Temporal-Difference Learning](https://arxiv.org/abs/1507.01569)：ETD 的 interest、follow-on 与线性收敛条件。阅读公式时区分当前与上一重要性比。

- [RLPark · Original reinforcement-learning implementations](https://github.com/rlpark/rlpark)：作者群体开发的线性与资格迹算法实现库。版本中的梯度 TD 命名和具体更新须逐项对应，不能只按类名互换。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 估计哪种策略的价值，又按什么状态分布加权？

行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

函数逼近与深度方法：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

持续学习中的研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](off-policy.md) → [数据与训练接口](../deep/practice.md) → [离线数据的覆盖](../deep/offline.md) → [流式更新](../../textbook/streaming.md)


[领域总览与问题地图](../../docs/field-framework.md) · [奖励假设与设计](../../textbook/reward-design.md) · [持续控制：完整学习器的比较](../../textbook/control.md)

- [通用价值函数与预测知识](../../textbook/gvf.md)
- [价值预测与资格迹](../../textbook/value.md)
- [深度价值学习](../../textbook/deep-value.md)

对应原始材料：第 11 章：Off-policy Methods with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
