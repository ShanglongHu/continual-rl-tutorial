# 作者源码接入验收 · 2026-10-04

`scripts/author_project.py prepare`真实抓取并detached checkout三个固定提交，随后verify核对commit、origin、clean状态与所有reading_order文件：

| 项目 | 固定commit | 阅读路径验收 |
|---|---|---|
| DreamerV3 | e01491fad6434b2245a3b8ca201dd7faedcc458c | 6/6 |
| TD-MPC2 | e9f59321933cbc8e11a002b842adc7d4ffae8ff1 | 7/7 |
| DiscoRL | 9059a29f7121d60948f25ef165e08e050e9399c8 | 7/7 |

本地源码位于gitignored `results/author-checkouts/{dreamerv3,tdmpc2,disco_rl}/`。上游源码没有复制进教程提交树。未安装依赖、未下载外部checkpoint/dataset、未运行训练或notebook。

已核对README及源码：Dreamer的Agent.loss/imag_loss/lambda_return与main.make_agent/make_env；TD-MPC2的TDMPC2.act/_plan/_update和WorldModel；DiscoRL的Agent.learner_step与DiscoUpdateRule.agent_loss。阅读顺序与真实运行入口登记在manifest。源码存在不是算法效果证据。

特别提醒：Dreamer defaults.run.steps为1e10；TD-MPC2默认steps为1千万且enable_wandb=true。这些默认不构成本次运行预算，完整训练前必须明确覆盖。DiscoRL要求Python3.11+，其两个notebook分别承担meta-evaluation/meta-training，不能将运行前者当作重现全部外层发现搜索。

3个行为测试通过：registry身份/入口、已有目录拒绝prepare、错误commit/dirty/remote/缺少阅读文件拒绝verify。没有作者工程依赖或训练验收；不能声称原论文精确结果已复现。
