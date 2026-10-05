# 当前论文研究线：固定原型系数可恢复性与原生预测

当前冻结主线（2026-09-30）：**在 YOLO26m-Seg 与 COCO 的既定实验范围内，建立固定原型下的系数可恢复空间与原生无 GT 系数预测之间的差距，并通过受控对照区分其可能形成环节；只有在机制获得证据后，才用针对性干预验证能否改善正常推理输出。**

当前阶段目标是收敛已有证据，区分“可恢复性事实”和“某种预测器的阴性结果”，定位原生系数预测链中尚未直接检验的环节。停止的是 `big-global` 外挂修正器支线；保留的问题是“有好系数，为什么原模型没有预测出来”。详见[主线冻结记录](../../../project/research_notes/MAINLINE_STATUS_20260930.md)。

历史 P3 干预实验、稿件和诊断仍保留在本研究线中，作为已完成证据或历史候选，不自动升级为当前机制结论。

## 当前论文文件

| 文件 | 用途 |
| --- | --- |
| [PAPER_DRAFT_ZH.md](PAPER_DRAFT_ZH.md) | 中文完整正文，便于讨论论述和方法 |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | 英文完整正文，含摘要、引言、相关工作、诊断、方法、实验、讨论、结论 |
| [EVIDENCE_MAP.md](EVIDENCE_MAP.md) | 每项主张与原始结果、实现、文献的对应关系；待补结果的具体位置 |
| [references.bib](references.bib) | 相关文献的 BibTeX |

状态：**完整研究初稿；当前机制主线已冻结，尚非最终投稿定稿。**

## 研究目录

后续工作根目录为 `C:\Dpan\codexproject\paper-disc-organized`。实验脚本放在各实验的 `scripts/`，新执行的日志、快照和结果放在该实验的 `runs/`；按根目录 AGENTS.md 直接落盘，由工作台自动扫描。

历史入口与证据见各实验目录；big-global 独立确认见 [7V 结果](../../../experiments/coefficient_frozen_confirmation_7V_20260929/RESULTS.md)。

## 历史入口与证据

以下文件继续保留，用于追溯 P3 干预路线和既有论文稿件，不代表当前已成立的方法结论：

- [P3 发展计划](DEVELOPMENT_PLAN.md)
- [证据地图](EVIDENCE_MAP.md)
- [中文历史稿](PAPER_DRAFT_ZH.md)
- [英文历史稿](PAPER_DRAFT.md)
- [最新双分支与 gate 归属诊断](../../../experiments/p3_branch_gate_diagnostic_20260915/REPORT.md)
- [Full COCO 一轮配对结果](../../../experiments/counterfactual_p3_fullcoco_20260914/RESULTS_20260915.md)
- [Pilot 方法实验](../../../experiments/counterfactual_p3_distillation_20260914/REPORT.md)
