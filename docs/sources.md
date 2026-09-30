# 来源、实现与材料边界

本仓库由 Ying Wen 的持续强化学习学习指南整理而来，发布为可独立阅读和运行的配套材料。
教程内容与代码在 Git 中按版本维护；复现时以所使用 commit 的命令和源码为准。

## 教学方法与基础

- [Sutton & Barto, Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：
  价值更新、控制、资格迹、规划与策略梯度的基础。
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/index.html)：
  参考“概念与推导 → 简洁实现 → 实验与阅读”的组织方式。
- [Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html)：
  实验设计、度量、随机性与调参的阅读入口。

本教程的数值例子、讲解和标准库教学脚本围绕本学习路线编写，
不是上述资料或深度研究仓库的逐字副本。

## 论文与外部实现

每章末尾列出对应原论文、作者代码或官方文档，并说明优先读什么。
进一步资料见[分方向阅读](advanced-reading.md)。
本仓库不打包第三方论文、视频、slides、模型权重、数据集或外部研究代码。

研究论文的结论应引用原论文。这里的代码用于机制学习；
例如标量 EWC 不等于估计神经网络 Fisher，手工记忆 oracle 不等于学出的 RNN。
研究代码可能需要旧版本依赖，应在独立环境中固定 commit 后运行。

## 许可

原创教学文字、解释、练习和图示采用 [CC BY 4.0](../LICENSE-DOCS.md)；
代码、嵌入式代码片段、测试与配置采用 [MIT](../LICENSE-CODE)。
第三方材料适用原许可，本仓库的许可不覆盖它们。
