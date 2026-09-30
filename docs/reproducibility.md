# 复现约定

[学习首页](../README.md) · [实验目录](experiments.md) · [协议模板](protocol.md)

## 两种运行不是同一种证据

- `python3 scripts/run_all.py --quick --out results/smoke`：短运行检查接口与输出，不用于比较方法。
- `python3 scripts/run_all.py --out results/teaching`：正文教学规模，帮助解释机制；不是充分调参后的科研排名。

两种模式都运行全部九组实验。标准库实现没有 GPU 或第三方包依赖。

## 一次运行留下什么

- 9 个 CSV：prediction、control、policy、dyna、consolidation、bandit、memory、credit、retention。
- 6 个 HTML：五组基础算法与 bandit。
- 1 个 manifest.json：命令、模式、Python、平台、Git commit/dirty 状态、代码和输出的 SHA-256、行数、完成状态。

`status=complete` 才表示整套命令完成；失败运行保留已有输出，不能作为完整实验。
manifest 中的命令在结果目录执行，脚本路径相对于仓库；最简单的重现方式仍是调用 `run_all.py`。
本脚本创建全新的结果目录，不会删除或覆盖已有目录。

## 默认完整教学规模

| 实验 | seeds | 长度 | 变化 |
|---|---:|---:|---|
| prediction | 5 | 1,000 episodes | 无 |
| control / policy / dyna | 各 5 | 各 800 episodes | 第 400 个 episode 后 |
| consolidation | 1 | 200 更新 | 标量目标对照 |
| bandit | 20 | 4,000 环境步 | 第 2,000 步后 |
| memory | 1 | 4,000 平衡试次 | 手工信息条件对照 |
| credit | 5 | 4,000 样本 | 固定信号与噪声 |
| retention | 1 | 先 1,000 步旧目标，再 1,000 步新目标 | 切换目标 |

随机实验使用 seeds=0…N−1。seed 相同不保证不同算法走相同轨迹；预测实验额外固定了相同输入轨迹。
memory / retention / consolidation 的重复不构成新的独立随机证据。

## 比较曲线前检查

1. 比较的是同样的环境步、episode、梯度更新，还是墙钟时间？Dyna 必须同时报真实交互和模型更新。
2. 真正终止与任意时间窗口结束是否区分？位置 reset 与参数 reset 是否区分？
3. 方法获得了相同的信息、数据、存储与调参预算吗？
4. 标准差不是置信区间；不要用单 seed 或最好的 seed 下结论。
5. 控制实验的窗口回报不是每个 episode 的完整轨迹日志；CSV 记录指标与配置，不宣称记录全部 transition。

自动测试检查已知数学例子、有限差分、等价特例、有限值、输出完整性和保护已有结果。
它不能证明某个方法在所有任务有效。

## 在 GitHub 上检查

[Teaching checks](https://github.com/ying-wen/continual-rl-tutorial/actions/workflows/test.yml)
使用 Python 3.10、3.12、3.13，运行单元测试与完整教学配置。
打开具体 workflow run 可以查看结果；成功运行的 CSV、HTML、manifest 以 artifact 保留 14 天。
徽章或某次通过只对应那个 commit，不自动证明未来代码也通过。

严格逐字节对照时同时固定 commit、Python 版本和平台；
跨版本浮点实现可能产生细微差别。科学结论还需要匹配任务和统计设计。
