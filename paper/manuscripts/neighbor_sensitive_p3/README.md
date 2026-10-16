# 原型—系数链路

当前研究线名称为 **“原型—系数链路”**（Prototype→Coefficient），研究固定预训练模型下的系数可恢复性与后训练实例修复。`neighbor_sensitive_p3` 保留为历史技术标识及目录名，用于连接既有论文和 Run；具体机制与方法按证据选择。

> **当前执行入口：** [统一科研总控](../../../project/ACTIVE_PLAN.md) → [GT 真实输出基线](../../../experiments/official_gt_failure_rebaseline_20261013/PROTOCOL.md) → [固定权重 Prototype→Coefficient 后训练计划](../../../experiments/mask_readout_mechanism_v2_20261012/POSTTRAIN_RESCOPE_20261013.md)。本页下文的“2026-09-30 历史主线记录”属于历史状态，不再授权 TAL 或五臂 20k。

**2026-10-17当前论文正文：** [结果与讨论](RESULTS_AND_DISCUSSION_ZH_20261017.md)围绕“可修机会—实际兑现—损伤代价”展开，逐项报告相对原生和已有方法的收益、损伤及未证边界，附[正文核验](RESULTS_AND_DISCUSSION_ZH_20261017.audit.json)。这是当前O2O后训练路线的正文节选；旧raw/argmax的MANUSCRIPT_ZH/main.tex与旧P3正文继续保留各自历史对象，尚未与本节合稿。

**论文证据入口：** [证据与主张表](EVIDENCE_AND_CLAIMS_20261017.md)，附[来源哈希及原始字段](EVIDENCE_AND_CLAIMS_20261017.sources.json)。本次只完成写作与证据核对，未追加训练或新实验。

## 历史主线记录（2026-09-30）

当时冻结的主线：**在 YOLO26m-Seg 与 COCO 的既定实验范围内，建立固定原型下的系数可恢复空间与原生无 GT 系数预测之间的差距，并通过受控对照区分其可能形成环节；只有在机制获得证据后，才用针对性干预验证能否改善正常推理输出。**

当时的阶段目标是收敛已有证据，区分“可恢复性事实”和“某种预测器的阴性结果”，定位原生系数预测链中尚未直接检验的环节。停止的是 `big-global` 外挂修正器支线；保留的问题是“有好系数，为什么原模型没有预测出来”。详见[主线冻结记录](../../../project/research_notes/MAINLINE_STATUS_20260930.md)。

历史 P3 干预实验、稿件和诊断仍保留在本研究线中，作为已完成证据或历史候选，不自动升级为当前机制结论。

## 当前论文文件

| 文件 | 用途 |
| --- | --- |
| [RESULTS_AND_DISCUSSION_ZH_20261017.md](RESULTS_AND_DISCUSSION_ZH_20261017.md) | 当前O2O后训练论文的中文“结果与讨论”正文；含双参照结果、解释与局限 |
| [EVIDENCE_AND_CLAIMS_20261017.md](EVIDENCE_AND_CLAIMS_20261017.md) | 当前后训练路线的论文主张、已证/未证边界、完整实验链路、双参照收益与研究取舍；非新实验 |
| [MANUSCRIPT_ZH.md](MANUSCRIPT_ZH.md) / [main.tex](main.tex) | 历史raw/argmax可恢复性完整稿及英文镜像；本轮未改写，不作为当前O2O方法结果正文 |
| [PAPER_DRAFT_ZH.md](PAPER_DRAFT_ZH.md) | 历史P3中文完整稿 |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | 历史P3英文完整稿 |
| [EVIDENCE_MAP.md](EVIDENCE_MAP.md) | 历史P3主张与原始结果、实现、文献的对应关系 |
| [references.bib](references.bib) | 相关文献的 BibTeX |

状态：**当前O2O“结果与讨论”正文已完成；引言与方法尚未按本次对象统一合稿。历史完整稿与当前章节分开保留，尚非最终投稿定稿。**

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
