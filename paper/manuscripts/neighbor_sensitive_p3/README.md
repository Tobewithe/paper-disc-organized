# 原型—系数链路

当前研究线名称为 **“原型—系数链路”**（Prototype→Coefficient），研究固定预训练模型下的系数可恢复性与后训练实例修复。`neighbor_sensitive_p3` 保留为历史技术标识及目录名，用于连接既有论文和 Run；具体机制与方法按证据选择。

> **当前执行入口：** [统一科研总控](../../../project/ACTIVE_PLAN.md) → [GT 真实输出基线](../../../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) → [固定权重 Prototype→Coefficient 后训练计划](../../../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md)。本页下文的“2026-09-30 历史主线记录”属于历史状态，不再授权 TAL 或五臂 20k。

## 历史主线记录（2026-09-30）

当时冻结的主线：**在 YOLO26m-Seg 与 COCO 的既定实验范围内，建立固定原型下的系数可恢复空间与原生无 GT 系数预测之间的差距，并通过受控对照区分其可能形成环节；只有在机制获得证据后，才用针对性干预验证能否改善正常推理输出。**

当时的阶段目标是收敛已有证据，区分“可恢复性事实”和“某种预测器的阴性结果”，定位原生系数预测链中尚未直接检验的环节。停止的是 `big-global` 外挂修正器支线；保留的问题是“有好系数，为什么原模型没有预测出来”。详见[主线冻结记录](../../../project/research_notes/MAINLINE_STATUS_20260930.md)。

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

## 2026-10-13：后训练修复主线（当前执行入口）

延续原研究命题：固定已训练的 YOLO26m-seg 官方权重，聚焦 Prototype→Coefficient 的后训练实例纠错。此前 raw/TAL/GT/COCO 的历史审计只是修正评价与证据身份，不代表方法主线转向 TAL；**历史 TAL 训练轨迹不是当前后训练研究的前置 Gate**。新执行从 [固定权重后训练重定向计划](../../../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md) 进入，先取得 [真实输出 GT 失败账本](../../../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md)，再对同一条最终 Box75 预测做固定原型 32D/实际 K 维系数有限 oracle，只有通过机会 Gate 后才准训练轻量 residual adapter。旧 10/12 五臂×20k TAL-E0 路由保留为未冻结历史草案，不再启动。
