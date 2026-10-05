# 两条论文研究线的引用关系与迁移决定

用户确认 paper-disc 中这两条线准备不同论文。按问题和证据分别推进，共享数据和代码通过来源关联。

实验独立存在，可以没有论文引用，也可以同时被多篇论文引用。下列论文线不是所有实验的强制分类；`research_lines.json` 中的 `study_ids` 表达引用关系，同一 Study ID 可以出现在多条线中。诊断也是实验，新建时统一使用 `experiments/`；旧目录按已记录的迁移决定保留兼容。

| 研究线 | 当前入口 | 主要执行位置 |
|---|---|---|
| 干预引导 P3 监督 | [论文 README](../paper/manuscripts/neighbor_sensitive_p3/README.md) | 本地研究/写作，56018 全量训练及后评估 |
| 预测修正收益的选择性掩码读出校准 | [论文 README](../paper/manuscripts/mask_boundary_calibration/README.md) | 本地稿件与实验主副本，28358lan 完整执行结果 |

第二条线已完成面向 Pattern Recognition Letters 的独立英文研究稿、补充材料及复现包，尚未投稿；论文引用既有实验 Study，不覆盖历史实验。第一条线保留原有身份和别名。机器可读归属见 [research_lines.json](research_lines.json)。

首次登记时的迁移决定（本次复制前的历史记录）：现有研究、稿件、运行与共享资产保持物理位置，完成论文线归属、缺失身份、主副本关系和历史版本登记。此前 244 条移动映射已核对，无路径断裂。旧目录刷新入口已转接新索引，停止多处手写总清单。

本地 baseline_s0/cfp3r_s0 一轮与远端同名三轮属于不同运行；两份本地 CCL 旧稿继续作为历史，不归入笔记本新论文。远端56018的 fullcoco、pilot、COCO 数据和旧 boundary 目录内的共享权重仍被在途任务引用。

首次盘点时，笔记本的 one-to-many 评价还会覆盖原 SUMMARY.json（以下为当时的历史状态，不是本稿当前运行）；协调任务已只读保存其旧内容快照，SHA256 为 `9fa5562ee47c5e9e7114a9bd263be9ec18cb7bad3f0a33b3c1bdfad58e7c0408`。早期盘点将旧 heldout 记为247图；负责任务已更正：远端 matched_records.csv 为4944图，旧脚本半集合为2472图，此前247来自本地500图副本的误读。仍不能作为符合4500图协议的验证。原摘要继续保留历史版本，更正来源见[复制交接记录](research_notes/COPY_MIGRATION_HANDOFF_20260915.md)。

运行结束后先核对退出状态、源码/协议版本、文件清单和指纹，再单独复制归档。任何旧目录都不会仅凭名称、seed或修改时间合并。具体逐目录决定、来源版本和后续条件见 [migration_decisions.json](../_maintenance/workbench_alignment/coordinator/migration_decisions.json)。

本次复制迁移的当前入口为 [START_HERE.md](../START_HERE.md)；上述首次盘点的执行状态不是实时状态。
