# paper-disc-organized

这是 `paper-disc` 科研项目的工作台镜像，目录结构以本地
`C:\Dpan\codexproject\paper-disc-organized` 为准。研究线不在本文件中固定，按 `project/research_lines.json`、论文文件夹和当前线程上下文发现。

仓库保存研究协议、实验登记、报告、脚本、论文材料和轻量运行元数据，便于在网页端查看研究进展。模型权重、COCO 数据集、图片、张量缓存、原始预测大表和传输压缩包保留在本地或计算服务器，不上传到 GitHub；相关实验中的路径、哈希和适用范围以协议及报告记录为准。

实验按 `experiments/<study>/` 组织，研究说明和跨实验判断位于 `project/`，论文材料位于 `paper/`，共享代码和轻量资产位于 `shared/` 与 `assets/`。

## 2026-10-12 候选身份与评价审计

已新增 [审计报告与证据清单](project/audits/candidate_eval_20261012/REPORT.md)、[本地最小重放交接](project/audits/candidate_eval_20261012/LOCAL_REPLAY_HANDOFF.md)、[跨研究线候选/评价强制规范](project/科研项目通用规范/候选身份与评估口径.md)。历史 Run 保留不覆盖，缺失的原始逐候选/权重/数据在本地校验，不得凭 GitHub 汇总值推断完整审计通过。

2026-10-12 补充：[指标最小化与 TAL-GT 分类规范](project/科研项目通用规范/指标最小化与TAL分类.md) 已取代“每个实验必报所有指标”的做法，并核实官方训练 TAL 与默认推理分支不是同一个评价过程。

本次审计补充：[指标最小化与 TAL-GT 失败分类](project/科研项目通用规范/指标最小化与TAL分类.md)、[只读 TAL→GT 归类脚本](project/audits/candidate_eval_20261012/summarize_gt_tal_cached.py)。这两份产物不把 TAL 回放当作 COCO 官方推理匹配，也不改写历史结果。

2026-10-13 [固定版本 O2O Top-K 和官方 COCO 基线核验](project/audits/candidate_eval_20261012/O2O_TOPK_OFFICIAL_COCO_BASELINE_20261013.md)：官方权重 SHA 完全一致，论文刊载 AP 与项目自定义评估存在约 0.5～0.6 点的协议差距，已列入通用规范的强制前置核验。

2026-10-13 [YOLO26m-seg 官方 COCO GT 失败重基线执行计划](experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md)：Study 已注册、状态 `planned_not_executed`；先原生基线 parity，后同一输出身份的逐 GT 失败分类，复用既有 raw 而不重训。
