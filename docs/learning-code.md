# 从一个算法文件到可追查的实验

本目录面向学习者。先读一个真实更新，再跟踪它如何进入完整经验流。所有结果来自运行；没有预置“应当上升”的曲线。

## 文件怎么分

```text
implementations/
  classic/        表格方法与线性预测：每个算法一个文件
  deep/           神经控制与离线学习：每个算法一个文件
  continual/      持续预测、元步长、保留、状态、平均奖励与规划
  runtime.py      参数、日志、失败记录、CSV 与 SVG；不包含算法更新
```

各组 `_common.py` 只共享环境、网络构造、采样和指标。对同一方法的关键变体，要看名称及源码中的假设。例如固定温度 SAC 不等于自动调温版本；单工作器同步 A2C 不等于分布式 A3C；已知奖励权重下的 GPI 不等于自动发现任务。

## 阅读顺序

1. 读 `META`：任务、可见信息、初始化、指标和预算时钟。
2. 读 `update`、target 函数或明确标记的更新段。确定每个量取更新前还是更新后的值。
3. 读 `run`：它在哪里采样、构造目标、写入参数、处理终止、评价和记录？
4. 看对应测试。手算、有限差分和退化等价回答实现问题；训练曲线回答该任务中的学习问题。
5. 再运行默认 baseline。不要先根据结果调整多个机制。

## 一条命令，保存完整结果

所有命令从仓库根运行。Python 3.10+；标准库方法不要求 GPU。深度方法另装 `examples/deep_requirements.txt` 中的 PyTorch。

```bash
python3 implementations/runtime.py --list
python3 implementations/classic/td_lambda.py --steps 1200 --seeds 0 1 2 3 4 --out results/td-lambda-first
python3 implementations/deep/ppo.py --steps 1200 --seeds 0 1 2 3 4 --out results/ppo-first
```

`--baseline none` 仅运行当前方法；`--baseline ALGORITHM_ID` 更换对照。运行器拒绝不同任务、指标、方向或预算时钟的直接比较。默认超参数都在算法文件中，修改代码后应使用新的输出目录和新的实验记录。

终端显示 `[prepare]`，然后显示当前算法、seed、阶段、当前时钟/预算与指标。phase 是测量时的任务或学习阶段，具体含义见源码。更新次数、TD error、loss、目标网络更新或模型备份等诊断随算法记录。它们不全是优化目标，也不保证单调下降。

输出结构：

```text
results/td-lambda-first/
  index.html                  可离线打开的对照图
  learning-curves.svg          矢量图
  curves.json                 均值、样本标准差、样本数
  manifest.json               配置、源码摘要、完整实验人口与失败
  td_lambda/seed-0/
    events.jsonl              原始阶段记录
    metrics.csv               逐记录点指标与诊断
  td0/seed-0/                 baseline，其他 seeds 同结构
```

预算时钟不总是环境步数。DP 使用模型扫描，三个离线 RL 例子使用固定512条转移上的训练批次（training_batches），递归预测例子可能使用序列。离线 Q-learning 与 CQL 每批各执行一次 Q 优化器更新，IQL 每批对 V、Q、actor 各执行一次，共三次；日志同时保留 training_batches 与 optimizer_steps。相同批次数不代表相同优化器调用数、梯度计算或计算成本。不要把它们放在“同样1000步”的全局排行榜。模型规划和评价还会消耗额外计算，需另查诊断和代码。

## 怎样读图

图显示全部指定训练种子的均值与 ±1 个样本标准差。五个种子是教学展示配置，不是统计充分性保证。没有置信区间或显著性检验。任务、参数和种子均公开；这些任务很小，有些算法很快都达到相同回报，平线也应保留。

先看完整曲线及每个种子。再问：差异来自目标估计、策略选择、学习率、模型备份还是保留旧样本？要做因果解释，应固定其他因素设计消融。对角线、缓慢学习和负结果不能删除。运行失败保存失败回执，不在成功种子里重新挑最好结果。

## 接入 Workbench

教程负责算法实现；Workbench 负责研究协议与证据管理。使用Python 3.10+，在 Workbench 仓库根运行，替换 `TUTORIAL_CHECKOUT` 为教程 checkout 的绝对路径。依赖环境应包含所选算法的依赖，并在plan/freeze/run/import阶段使用同一解释器。

有Workbench checkout时直接进入该目录；源码根的 `python3 -m ...` 不要求安装包。没有基本仓库时可先获取：

```bash
git clone https://github.com/ying-wen/rl-research-workbench.git
cd rl-research-workbench
```

**版本前提：**使用包含 `rlworkbench/tutorial_adapter.py` 的 Workbench 版本。已有较早 checkout 的读者应先更新；运行 `python3 -m rlworkbench.tutorial_adapter --help` 确认接入命令可用。

若需要安装Workbench，推荐在该checkout建立隔离环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

bandit示例无需第三方依赖；深度方法先在同一环境安装教程 `examples/deep_requirements.txt`，再建立协议和冻结版本。修改Python/包版本或算法源码后用新协议、新锁与新输出目录。完整接入说明见Workbench `docs/tutorial-adapter.md`。

```bash
python3 -m rlworkbench.tutorial_adapter plan \
  --tutorial-root TUTORIAL_CHECKOUT \
  --ids bandit_constant_step bandit_sample_average --baseline bandit_sample_average \
  --study-id bandit-teaching-v1 --seeds 11 23 --holdout-seeds 101 211 \
  --steps 1000 --failure-score -10 --out studies/bandit-teaching-v1.json
python3 -m rlworkbench freeze studies/bandit-teaching-v1.json --external --out studies/bandit-teaching-v1.lock.json
python3 -m rlworkbench.tutorial_adapter run studies/bandit-teaching-v1.lock.json \
  --tutorial-root TUTORIAL_CHECKOUT --out incoming/bandit-teaching-v1
python3 -m rlworkbench import-results studies/bandit-teaching-v1.lock.json \
  --input incoming/bandit-teaching-v1 --out runs/bandit-teaching-v1
python3 -m rlworkbench audit runs/bandit-teaching-v1
python3 -m rlworkbench report runs/bandit-teaching-v1
python3 -m rlworkbench compare runs/bandit-teaching-v1 --candidate bandit_constant_step --baseline bandit_sample_average
```

适配器冻结源码摘要并拒绝漂移；保存完整算法×种子人口。`failure-score` 是预先规定的复合分析分值，不是算法真实回报。对 RMSE 等越小越好的指标，应设定高而非低的失败分值。外层不重复评价；使用算法自己记录的指标，因此先读指标定义。

此流程不自动启动 HPO、分布式训练或检查点恢复。大型作者工程保留原训练器，参见[作者工程接入](author-projects.md)和 Workbench 的 `docs/external-adapters.md`。

## 维护网页结果

源码确认后运行一次固定小预算；同一算法在每个seed只训练一次，其原始记录用于不同baseline图。生成的是可重建的快照，不手工编辑曲线。

```bash
python3 scripts/build_learning_gallery.py --out results/gallery-v1 --steps 1200 --seeds 0 1 2 3 4
python3 scripts/build_learning_gallery.py --out results/gallery-v1 --export-only --site WEBSITE_CHECKOUT
```

导出核对源码摘要、数据完整性和可比条件。修改代码后需要新运行。原记录不覆盖。网站只复制明确列出的教学源码与结果，不复制作者 checkout、凭据或本机环境。
