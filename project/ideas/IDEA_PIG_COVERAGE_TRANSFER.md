---
schema_version: 1
id: IDEA_PIG_COVERAGE_TRANSFER
kind: idea
title: 从同场景拟合走向新条件可靠性：覆盖、迁移与保持
summary: 研究可迁移的实例能力与可更新场景信息；有限成像首试已收束，新条件覆盖和安全适配仍未知。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_REAL_DATA, OUTLINE_PIG_DATA, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: paused
assessment: unknown
author: AI依据用户澄清及数据增强、迁移文献作开放构思
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: Copy-Paste、合成猪检测、SAP与DAFormer的迁移条件和限制
    relation: references
---

## 想法与价值

相机、猪栏、照明、品种和生长阶段改变时，区分需要保留的实例能力与可少量更新的场景信息。真实条件覆盖、二维重组、合成变化、少量目标弱标、未标视频及适配均可成为途径，选择取决于实际缺失信息。

固定相机的空栏或长期录像也可能提供设施与活动区域上下文，但稳定视角和参考可得性尚未核实。与[关系标注](IDEA_PIG_RELATION_ANNOTATION.md)、[受控合成](IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)的联系在于辨别哪些信息值得获得，具体方法不预定。

## 当前判断与依据

本Idea暂缓、判断未知。[有限合成release](../../experiments/spm_concept_synthesis_repair_20261101/REPORT.md)的资产家族与来源覆盖有限；反复使用开发集也未建立跨农场结论。

[成像首试](IDEA_PIG_APPEARANCE_STABILITY_V1.md)未支持本组变化的大幅粗粒度损失，细边界、排序与新条件风险仍开放。[夜间保持](IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)缓解了相对训练退化，但未形成可采用模型。固定模型稳定性、更新损伤和新域迁移因此继续分别判断。

## 值得补充的依据

优先需要明确的新来源、缺失条件、目标标签或细粒度失效证据，再比较覆盖与适配方案。原“C/Q×覆盖”的局部候选只回答语义边际作用，其他观察、弱标和场景适配可按各自问题发展。已有标签权限、拆分与封存测试继续沿原协议。

探索由[条件可靠性主线](../../paper/manuscripts/pig_condition_reliability/README.md)组织，行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_COVERAGE_TRANSFER.md)保留候选展开、首试与综合反馈。
