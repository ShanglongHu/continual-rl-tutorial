[学习首页](../README.md) · [算法与代码目录](README.md)

# 10 每步都要及时更新：TD(λ)、IDBD、平均奖励与 Stream-X

先修：TD、actor–critic、向量梯度

问题：没有 replay 和大 batch，怎样在每个新样本后稳定地更新？

增量 TD / traces → 逐参数步长 → 平均奖励目标 → 深度 streaming 实现

Streaming 约束数据与更新方式；average reward 改变优化目标；single-life 约束环境重置。这三个概念互不等价。先用线性预测理解迹与步长，再看深度 streaming 代码如何处理输入、奖励尺度和更新幅度。

符号：xᵢ：特征；wᵢ：权重；δ：预测误差；βᵢ=log αᵢ；hᵢ：权重对 log 步长的近似敏感度；μ：meta 步长；r̄：平均奖励估计。

## 资格迹解决时间信用，不是自动步长

e 累积过去特征的责任；α 决定给定责任下实际更新多少。二者可以一起使用，但一个大的 trace 也可能放大更新，所以需要看特征尺度与控制方法。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

## IDBD：不同参数能否自动学得快慢不同？

在线性监督预测中，让每个参数有自己的 log 步长，利用误差与过去更新敏感度调节 β。下面是带非负因子的常见线性近似：先更新 β，再算 α、w 和 h，所有右侧误差来自更新前预测。

$$
\begin{aligned}\beta_i&\leftarrow\beta_i+\mu\delta x_i h_i,\quad\alpha_i=\exp(\beta_i)\\w_i&\leftarrow w_i+\alpha_i\delta x_i\\h_i&\leftarrow h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i\end{aligned}
$$

现有 credit 教学实验还对 β 做了数值裁剪。这不是 NetworkIDBD，也不能不改推导就用于所有 TD/actor–critic 目标。

## 长期运行也可以用平均奖励目标

折扣目标不是唯一选择。平稳、适当遍历条件下，可研究单位时间平均奖励及相对价值；其 TD 误差减去平均奖励估计。非平稳世界通常还需要有限窗口或动态目标，不能直接假定单一稳态平均存在。

$$
\bar r^\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\sum_{t=1}^TR_t,\qquad\delta_t=R_{t+1}-\widehat{\bar r}_t+v(S_{t+1})-v(S_t)
$$

## 从线性更新进入 Stream-X

阅读官方实现时分开检查归一化、初始化/表示、梯度与参数更新控制，以及 actor–critic 顺序。省去 replay 与 target network 以后，这些组件会改变训练动力学；删除 buffer 本身不是完整方法。

## 手算例子

x=1、δ=2、旧 h=0.5、μ=0.01，则 β 增加 0.01，步长乘以 exp(0.01)≈1.01005。它是“小幅调整未来学习速度”，不是把当前权重直接加 0.01。

## 对照代码

```python
# 已有 crl_labs.py 的 credit()：先从线性预测理解
beta[i] += meta * delta * x[i] * h[i]
rate = exp(beta[i])
w[i] += rate * delta * x[i]
h[i] = h[i] * max(0, 1 - rate*x[i]*x[i]) + rate*delta*x[i]
# 深度 actor–critic 请回到完整实现，不直接粘贴此监督更新
```

先读 crl_labs.py 的 credit()；再进入 Stream-X 的 stream_ac_discrete.py、optimizer.py 和 obs_reward_transforms.py，逐段追 act → transition → update。平均奖励实现另看 Naik 等的仓库，不与 streaming 当作同一算法。

```sh
python3 examples/crl_labs.py credit --seeds 5 --steps 4000 --out credit-study.csv
```

改变干扰特征数和噪声，观察信号误差与各类特征步长。SGD 与 IDBD 都要获得合理调参预算；固定一个默认步长的差异不能直接证明自动步长普遍更好。

## 怎样接到 CRL

这是通向 Oak 的选择性学习、在线 meta-gradient、Stream-X、实时递归学习的基础接口。先明确你改的是信用、尺度、参数步长还是更新预算。

适用条件：线性 IDBD、平均奖励 TD 和深度 Stream-X 的条件不同。本章把它们连成学习路线，不宣称共享同一个收敛定理。

## 思考题

一个算法没有 replay，但每次失败后 reset，能否称为 single-life？

提示：数据权限与环境权限分别检查。

参考答案：不能仅凭无 replay 判断。它可满足 streaming 更新协议，却仍然使用 episodic reset；single-life 需要另外说明失败与恢复机制。

## 原始材料与代码

- [IDBD 原论文](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)：读线性步长的敏感度近似。
- [Stream-X 官方代码](https://github.com/mohmdelsayed/streaming-drl)：从离散动作单文件和 optimizer 开始。
- [平均奖励方法代码](https://github.com/abhisheknaik96/average-reward-methods)：用小任务区分平均奖励与折扣价值。


---

[← 上一章](09-plasticity.md) · [下一章 →](11-state.md)
