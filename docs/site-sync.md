# 网页与仓库的一致性

基础分册、CRL 教材、实验手册和教学 Python 源文件由网站的结构化内容生成。仓库提供 Markdown 阅读入口与可直接运行的目录布局。代码按原字节复制；Markdown 只调整已知的内部链接和命令路径。

教材嵌入的模板化原创 SVG 同步到 `assets/crl-figures/`，正文使用相对路径；存在的手机版也一并保留。这些图无需访问网站即可阅读。第三方原图、未纳入导出的旧图、位图、数据文件及外部论文仍保留线上入口，不属于完整离线资源包。

## 读者怎样核对

- `data/site-export.json` 列出每个公开下载文件的原始 SHA-256、仓库目标路径和章节映射。
- `data/site-sync.json` 列出转换后的文件摘要。它不是数字签名，只检查意外漂移。
- CI 检查原样代码的字节一致性、正文完整性、链接和命令路径，并执行教学测试。
- 网站的 GitHub 链接指向具体教程 commit。不要把主分支未来的新内容当作旧实验实际使用的源码。

## 维护者同步

先在网站仓库构建。将公开的 `download` 输出目录传给同步工具：

```bash
python3 scripts/sync_site.py --source-directory ../yingwen-site/dist/zh/continual-rl/download
python3 scripts/sync_site.py --source-directory ../yingwen-site/dist/zh/continual-rl/download --check
python3 -m unittest discover -s tests -v
python3 scripts/test_examples.py
```

同步工具先检查源文件摘要。它只写清单中的安全相对路径，并拒绝覆盖已改动的生成文件。修改正文或公式时，先修改网站维护源，再重新导出。README、工作流和同步工具本身独立维护。

图片也按原字节与摘要同步，不在仓库中维护第二套绘图源。需要修改图时，修改网站对应生成器并重新生成、构建、同步。原创教程图适用 CC BY 4.0；不以这项许可替换第三方资产各自的许可。

旧版十三个 `algorithms/` 路径保留为完整章节副本。新版正文目录是 `textbook/`；基础分册在 `foundations/`。八课 CRL 导论仍放在 `tutorials/`。

公开索引收录的论文、作者代码和 benchmark 保留各自许可证。本仓库不复制或重新许可外部论文全文、ROM、数据集或第三方完整代码库。
