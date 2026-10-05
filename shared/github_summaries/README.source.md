# paper-disc：实验概述与证据索引

这是科研项目的**私有文档快照**，供 GitHub 和网页端 ChatGPT 阅读。主要研究对象是官方 COCO 预训练 YOLO26m-seg；具体版本、分支、候选、损失与解码范围以每份实验报告为准，历史结果不自动适用当前协议。

## 从哪里开始

1. [当前研究状态](CURRENT_STATUS.md)：已建立事实、停止分支、未解决问题及不同协议的区别。
2. [全部实验索引](EXPERIMENT_INDEX.md)：{{STUDIES}} 个稳定 Study 单元，含历史导入、方法、诊断、消融、失败尝试和冒烟测试。
3. [机器可读目录](catalog.json)：Study ID、类型、登记状态、报告入口和已登记 Run。
4. [证据来源版本](SOURCE_MANIFEST.json)：源文件的项目相对位置与 SHA256；不包含账号、连接信息、数据或权重。

`experiments/` 是每个单元的概述，`reports/` 是对应现有报告及小型统计快照。大规模逐实例数据、数据集、权重、代码执行命令、日志和机器连接凭据留在科研项目。**这里没有重新运行实验。**

## 给 ChatGPT 的阅读请求示例

> 请读取 GitHub 仓库 `Tobewithe/paper-disc-summary` 的 `README.md`、`CURRENT_STATUS.md` 和 `EXPERIMENT_INDEX.md`。先汇总已经做过的实验，再分析下一步。凡是认为需要新实验，请先检查索引是否已有同一问题的证据。区分 raw 候选、官方 TAL 正样本、几何匹配候选和最终推理输出；不要混用其数字。引用对应报告路径，指出结论范围、阴性结果和未解决的信息。

指定实验时，可直接说：

> 读取 `experiments/coefficient_official_tal_affine_20260930.md` 及其链接报告，解释共享末层拟合和新图收益的区别。

## 阅读时的限制

- 一个登记单元不等于一个已完成且独立的科学结论。`imported` 只表示历史导入；缺少结果或状态时保持未知。
- 概述中的“源报告节选”是原作者文本，不是本次对全部历史实验的重新验证；修正报告和当前范围说明优先于过期解释。
- GT 辅助 oracle、理想修复和输入编辑诊断不等于正常无 GT 方法收益；候选 IoU、Mask75 与 COCO AP 不相同。
- 图片/候选数量、统计单位、one-to-one/one-to-many 分支、不同标签或损失口径必须匹配，才能比较实验。
- 新图片若已参与历史方案选择，不能重新称为盲测。当前仍没有建立主线上的强方法收益或唯一失败机制。

## 更新方式

这份仓库由科研项目中的 `shared/github_summaries/` 发布工具生成。更新现有实验报告后，在 PowerShell 执行：

```powershell
& 'C:\Dpan\codexproject\paper-disc-organized\shared\github_summaries\publish_summaries.ps1' -Push
```

脚本先重建快照、执行凭据与链接检查，再提交并推送到 `main`；没有变化时不会产生空提交。GitHub 不会自动读取本机新增文件。`PUBLICATION.json` 记录当前快照的导出时间和覆盖数量。

仓库保持私有。网页端 ChatGPT 需要使用已连接且获准访问这个仓库的 GitHub 账号；终端 GitHub 登录与 ChatGPT 连接授权是两件事。
