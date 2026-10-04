# 正文算法实现覆盖盘点（2026-10-04）

覆盖23章主教材、26章基础分册与8课导论；编号正文中的具体方法纳入，参考文献标题本身不构成新增算法。导论重复内容合并。本次实际发现 59 个独立教学模块，包含教学基线、预测/规划与控制实现；不能统称59个完整RL方法，更不代表原论文重训。

机器清单：`integrations/algorithm_coverage.json`。状态为 `independent_implementation / formula_core / author_project / unintegrated`，逐条保留章节、独立ID或综合lab函数/固定作者源码入口。独立文件存在与真实全人口运行完成分开：后者须检查运行manifest，不由本表认证。不计算百分比覆盖率。

## 已有独立教学文件

| ID | 方法 | 实际范围 | 对照与任务 |
|---|---|---|---|
| `bandit_constant_step` | 固定步长 ε-greedy | teaching-control | `bandit_sample_average`；stationary_bernoulli_bandit |
| `bandit_gradient` | 梯度赌博机 | teaching-control | `bandit_sample_average`；stationary_bernoulli_bandit |
| `bandit_sample_average` | 样本均值 ε-greedy | teaching-control | `bandit_constant_step`；stationary_bernoulli_bandit |
| `bandit_ucb` | 置信上界 UCB | teaching-control | `bandit_sample_average`；stationary_bernoulli_bandit |
| `double_q` | Double Q-learning | teaching-control | `q_learning`；chain_control |
| `dyna_q` | Dyna-Q | teaching-control | `q_learning`；chain_control |
| `emphatic_td` | Emphatic TD(0) | teaching-prediction | `linear_td`；random_walk_linear_prediction |
| `expected_sarsa` | Expected SARSA | teaching-control | `q_learning`；chain_control |
| `gtd2` | GTD2 | teaching-prediction | `linear_td`；random_walk_linear_prediction |
| `linear_td` | 线性半梯度 TD | teaching-prediction | `gtd2`；random_walk_linear_prediction |
| `mc_prediction` | 首次访问蒙特卡洛预测 | teaching-prediction | `td0`；random_walk_prediction |
| `nstep_sarsa` | 三步 SARSA | teaching-control | `q_learning`；chain_control |
| `nstep_td` | 三步 TD 预测 | teaching-prediction | `td0`；random_walk_prediction |
| `policy_iteration` | 策略迭代 | exact-planning | `value_iteration`；chain_exact_planning |
| `prioritized_sweeping` | 优先扫描 | teaching-control | `q_learning`；chain_control |
| `q_learning` | Q-learning | teaching-control | `sarsa`；chain_control |
| `sarsa` | SARSA | teaching-control | `q_learning`；chain_control |
| `sarsa_lambda` | SARSA(λ) | teaching-control | `q_learning`；chain_control |
| `td0` | TD(0) | teaching-prediction | `mc_prediction`；random_walk_prediction |
| `td_lambda` | 累积迹 TD(λ) | teaching-prediction | `td0`；random_walk_prediction |
| `tdc` | TDC | teaching-prediction | `linear_td`；random_walk_linear_prediction |
| `true_online_td` | True-online TD(λ) | teaching-prediction | `td0`；random_walk_prediction |
| `value_iteration` | 价值迭代 | exact-planning | `policy_iteration`；chain_exact_planning |
| `chain_q` | 链任务 Q-learning | teaching-control | `potential_shaping`；potential_chain_control |
| `constant_step_lms` | 固定步长 LMS | teaching-prediction | `idbd`；adaptive_regression |
| `count_bonus` | 计数探索奖励 | teaching-control | `plain_q_exploration`；count_exploration_chain |
| `differential_q` | 差分 Q-learning | teaching-control | `differential_sarsa`；continuing_bandit_control |
| `differential_sarsa` | 差分 SARSA | teaching-control | `differential_q`；continuing_bandit_control |
| `ewc` | 对角 EWC | teaching-prediction | `online_sgd`；retention_regression |
| `gvf_gtd_lambda` | 向量 GVF GTD(λ) | teaching-prediction | `gvf_td`；vector_gvf_prediction |
| `gvf_td` | 向量 GVF TD | teaching-prediction | `gvf_gtd_lambda`；vector_gvf_prediction |
| `idbd` | IDBD | teaching-prediction | `constant_step_lms`；adaptive_regression |
| `learned_model_mpc` | 学习模型滚动规划 | teaching-control | `one_step_model`；learned_chain_model_control |
| `one_step_model` | 学习模型一步贪心 | teaching-control | `learned_model_mpc`；learned_chain_model_control |
| `online_sgd` | 在线 SGD | teaching-prediction | `ewc`；retention_regression |
| `option_smdp_q` | SMDP 选项 Q-learning | teaching-control | `primitive_q`；option_chain_control |
| `plain_q_exploration` | 无探索奖励 Q-learning | teaching-control | `count_bonus`；count_exploration_chain |
| `potential_shaping` | 势函数奖励塑形 | teaching-control | `chain_q`；potential_chain_control |
| `primitive_q` | 原始动作 Q-learning | teaching-control | `option_smdp_q`；option_chain_control |
| `reservoir_replay` | 蓄水池经验回放 | teaching-prediction | `online_sgd`；retention_regression |
| `rtrl` | RTRL | teaching-prediction | `tbptt`；recurrent_sequence_prediction |
| `successor_features_gpi` | 后继特征 GPI | teaching-control | `successor_features_policy`；successor_feature_transfer |
| `successor_features_policy` | 后继特征固定基础策略 | teaching-control | `successor_features_gpi`；successor_feature_transfer |
| `tbptt` | 截断 BPTT | teaching-prediction | `rtrl`；recurrent_sequence_prediction |
| `tidbd` | TIDBD(λ) | teaching-prediction | `constant_step_lms`；adaptive_regression |
| `deep-a2c` | A2C | teaching-control | `deep-vpg`；deadline-chain |
| `deep-c51` | C51 | teaching-control | `deep-dqn`；deadline-chain |
| `deep-cql` | CQL | teaching-control | `deep-offline_q_learning`；offline-deadline-chain-fixed512 |
| `deep-ddpg` | DDPG | teaching-control | `deep-td3`；bounded-lq |
| `deep-double_dqn` | Double DQN | teaching-control | `deep-dqn`；deadline-chain |
| `deep-dqn` | DQN | teaching-control | `deep-double_dqn`；deadline-chain |
| `deep-iql` | IQL | teaching-control | `deep-offline_q_learning`；offline-deadline-chain-fixed512 |
| `deep-offline_q_learning` | Offline Q-learning | teaching-control | `deep-cql`；offline-deadline-chain-fixed512 |
| `deep-ppo` | PPO | teaching-control | `deep-vpg`；deadline-chain |
| `deep-qr_dqn` | QR-DQN | teaching-control | `deep-dqn`；deadline-chain |
| `deep-sac` | SAC | teaching-control | `deep-ddpg`；bounded-lq |
| `deep-td3` | TD3 | teaching-control | `deep-ddpg`；bounded-lq |
| `deep-trpo` | TRPO | teaching-control | `deep-vpg`；deadline-chain |
| `deep-vpg` | VPG | teaching-control | `deep-a2c`；deadline-chain |

关键边界：TIDBD教学γ=0退化为监督即时预测，未用此曲线证明带bootstrap的TIDBD；EWC/replay是监督保留回归，不能作为RL方法完整有效性证据；SF/GPI奖励权重直接告知，未学习奖励权重；RTRL/TBPTT是小递归预测；Deep方法是DeadlineChain/有界LQ/固定离线小数据教学训练。每个模块META/中文步骤写明目标、计步、初始化、评价与限制。

## 仅综合lab公式或子机制

| 方法 | 实际源码范围 |
|---|---|
| 迭代策略评价 | `examples/tabular_textbook_lab.py`：evaluate_policy |
| Every-visit MC | `examples/tabular_textbook_lab.py`：mc_visits |
| MC control | `examples/tabular_textbook_lab.py`：mc_control；已有综合lab小任务循环，尚无独立文件 |
| Ordinary IS | `examples/tabular_textbook_lab.py`：is_estimates |
| Weighted IS | `examples/tabular_textbook_lab.py`：is_estimates |
| Tree Backup | `examples/tabular_textbook_lab.py`：qsigma_target sigma=0；目标核不是完整控制训练 |
| Q(σ) | `examples/tabular_textbook_lab.py`：qsigma_target |
| Gradient MC | `examples/approximation_textbook_lab.py`：gradient_mc |
| LSTD | `examples/approximation_textbook_lab.py`：lstd |
| 半梯度 SARSA | `examples/approximation_textbook_lab.py`：semi_gradient_sarsa |
| Differential TD | `examples/lifelong_algorithms_lab.py`：differential_td |
| Relative Value Iteration | `examples/lifelong_algorithms_lab.py`：rvi_sweep |
| 平均奖励 SMDP option 更新 | `examples/lifelong_algorithms_lab.py`：option_rate_step |
| Watkins Q(λ) | `examples/credit_assignment_lab.py`：control_trace_step |
| Retrace | `examples/credit_assignment_lab.py`：offpolicy_trace_coefficients |
| V-trace | `examples/credit_assignment_lab.py`：vtrace_targets |
| Greedy adaptive λ | `examples/credit_assignment_lab.py`：greedy_lambda；已知bias/variance下局部最优，不是完整统计量学习 |
| Expected Eligibility Traces | `examples/credit_assignment_lab.py`：expected_trace_enumeration；枚举核验，没有学习期望迹网络 |
| Gradient Eligibility Traces | `examples/credit_assignment_lab.py`：gradient_trace_equivalence；非完整非线性控制训练 |
| REINFORCE | `examples/approximation_textbook_lab.py`：reinforce_gradient |
| GAE | `examples/deep_textbook_lab.py`：gae；独立deep训练器也复用GAE，组件不是独立控制算法 |
| Thompson Sampling | `examples/extended_foundations_lab.py`：thompson_action；后验采样核 |
| Per-decision IS | `examples/extended_foundations_lab.py`：per_decision_is |
| Sequential Doubly Robust | `examples/extended_foundations_lab.py`：sequential_dr |
| CMDP primal-dual | `examples/extended_foundations_lab.py`：primal_dual_step |
| 2×2 minimax | `examples/extended_foundations_lab.py`：minimax_2x2；解析矩阵核 |
| COMA counterfactual advantage | `examples/extended_foundations_lab.py`：counterfactual_advantage；不是完整COMA训练 |
| QMIX monotone mixing | `examples/extended_foundations_lab.py`：monotone_joint_greedy；不是QMIX神经训练 |
| Bayes filter | `examples/state_meta_lab.py`：belief_step |
| 完整 BPTT | `examples/state_meta_lab.py`：bptt_final；固定参数导数参考，不是通用RNN训练 |
| RTU | `examples/state_meta_lab.py`：rotation_trace；固定旋转单元敏感度，不是完整RTU actor-critic |
| Metatrace | `examples/meta_frontier_lab.py`：LinearMetatrace；critic-only unnormalized scalar core |
| Meta-gradient RL | `examples/state_meta_lab.py`：lambda_return_sensitivity；return参数导数核 |
| MAML | `examples/state_meta_lab.py`：scalar_maml；标量二次任务，不是MAML-RL完整训练 |
| ObGD 2024 | `examples/lifelong_algorithms_lab.py`：td_lambda_step；2024缩放核，不是2026版本 |
| Intentional Updates | `examples/meta_frontier_lab.py`：IntentionalStep；完整优化器默认时序核，缺actor-critic环境主循环 |
| CLEAR | `examples/lifelong_algorithms_lab.py`：clone_loss/vtrace_target；缺新旧经验采样与完整策略训练 |
| ReDo | `examples/lifelong_algorithms_lab.py`：redo_indices/recycle_one_unit；神经替换小诊断 |
| Continual Backpropagation | `examples/lifelong_algorithms_lab.py`：cbp_candidates/cbp_contribution_utility；神经替换小诊断 |
| Random Network Distillation | `examples/lifelong_algorithms_lab.py`：rnd_step；标量预测核，不是随机神经目标训练 |
| Learning progress curriculum | `examples/lifelong_algorithms_lab.py`：progress_score；单步分数核 |
| Recovery filter | `examples/lifelong_algorithms_lab.py`：recovery_filter；已知恢复概率过滤核 |
| UVFA 目标条件价值 | `examples/knowledge_algorithms_lab.py`：goal_q_update；表格目标条件核，不是通用神经UVFA |
| Hindsight Experience Replay | `examples/knowledge_algorithms_lab.py`：future_her/relabel_goal；缺完整goal-conditioned replay训练 |
| Subtask stopping | `examples/knowledge_algorithms_lab.py`：subtask_td/stop_rules |
| Intra-option Q-learning | `examples/knowledge_algorithms_lab.py`：intra_option_update |
| Option-Critic | `examples/knowledge_algorithms_lab.py`：option_critic_actor_step；动作/终止梯度核 |
| DIAYN skill information reward | `examples/knowledge_algorithms_lab.py`：skill_intrinsic_rewards；非完整DIAYN发现训练 |
| Option reward/endpoint model | `examples/knowledge_algorithms_lab.py`：model_td_update |
| Option Value Iteration | `examples/knowledge_algorithms_lab.py`：option_value_iteration；综合lab可解小模型规划 |
| Bradley–Terry preference reward | `examples/reward_design_lab.py`：preference_loss_gradient |
| MaxEnt IRL | `examples/reward_design_lab.py`：maxent_loss_gradient；有限轨迹手算核 |
| Intrinsic reward meta-gradient | `examples/reward_design_lab.py`：intrinsic_meta_gradient |
| Reward Centering | `examples/lifelong_algorithms_lab.py`：centered_single_state_step/centered_single_state_fixed_point |

这些条目尚无完整独立方法；部分综合lab已有小任务循环，但单步目标、等价核验、标量导数或数值投影不等同算法训练。GAE、Bayes filter等是计算组件，本表追踪其实现而不将其另算完整RL控制方法。

## 作者工程入口：源码已接入，未完整重训

| 方法 | 固定接入身份 |
|---|---|
| DreamerV3 | 作者维护reimplementation；固定源码与入口已核验，未安装训练；详见 `integrations/author_projects.json` |
| TD-MPC2 | 官方项目固定源码已核验，未训练；详见 `integrations/author_projects.json` |
| DiscoRL | minimal JAX harness meta-eval/meta-train；不包含原始全部搜索系统，未训练；详见 `integrations/author_projects.json` |

三个固定提交已真实prepare并verify；源码在gitignored results/author-checkouts，未纳入教程版权主树。无依赖安装、外部数据/权重下载或训练验收。Dreamer的Dyna教学连接与DiscoRL的IDBD入门不是对应完整系统。运行入口/预算及实际验收见 `docs/author-projects.md` 和 `docs/author-projects-validation-2026-10-04.md`。

## 正文尚未接入的方法

UCT / MCTS；RVI Q-learning；Stream-X / Streaming actor–critic；StreamingOptimizer 2026；GVFN；GRU / LSTM 通用状态网络；UORO；LRU；RL²；PEARL；Learned Policy Gradient；DRAGO；NaP；C-CHAIN；Parseval regularization；LSAC；HIQL；MaestroMotif；Eigenoptions；ROD；DCEO；ALLO；METRA；Laplacian Keyboard；Default Representation；ALPS；CPO。

这些方法保留缺口，不生成虚构实测图。后续接入应在新的协议/版本登记目标、数据权限、依赖、预算、基线与完整失败人口。

## 验收纪律

独立文件、公式测试、真实小任务训练、多seed教学对照、原论文复现是不同状态。结果图必须从已保存原始记录与完整种子人口复算，不能按成功前缀作图；旧任务保留误差与新任务适应结果同时检查。源码或META改变需新运行目录，旧证据保留。SGD/Adam/RMSProp/Welford、共轭梯度、Polyak、LayerNorm、重要性比率与tanh密度校正为组件，应在相应算法中说明，不冒充完整RL方法。
