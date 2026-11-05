---
schema_version: 1
id: IDEA_PIG_APPEARANCE_STABILITY_V1
kind: idea
title: 固定几何时，成像外观变化是否破坏实例区分？
summary: 有限成像干预未达到预定的大幅粗粒度损失，本版关闭；细边界、排序和新条件风险仍开放。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_DATA, OUTLINE_PIG_TASK, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: closed
assessment: contradicted
author: AI依据用户从剩余缺口选择并完成首个验证实验的指令形成
curation_status: source_based_draft
scope:
  claim: 当前原生模型在几何和标签不变的成像变化下，会丢失足够多可同时区分的实例，值得继续研究条件保持。
  stage: completed_large_effect_below_frozen_size
  tested_scope: 协议V1的真实损失至少5%及合成差异至少10%预测，限定低阈值正常输出的160格点IoU50计数。
  limits: 真实TRAIN数字色调与有限合成照明分别解释；非方法效用、非独立泛化和背景捷径证明。
evidence:
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 完整冻结样本、真实0.136986%及合成0的主量、区间和未达效应线判断
    relation: references
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 30个几何组的A/B灯光对照及辅助标签资格，身体／环境各一家族
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 跨条件、覆盖和域适配提供相关问题依据，不代表本假说已成立
    relation: references
---

## 本版主张与选择

固定模型、几何和标签，检验成像变化是否造成足够大的实例区分损失，为条件保持研究提供依据。真实TRAIN数字色调与现有合成物理照明分别比较，剂量、预选样本、效应线及停止条件见[协议](../../experiments/pig_appearance_stability_20261103/PROTOCOL_V1.md)；选择这一首试的背景见[设计说明](../../experiments/pig_appearance_stability_20261103/DESIGN_AND_SELECTION.md)。

## 当前判断与范围

本版关闭，预定的大幅粗粒度损失预测未获支持。两侧都有足够基线支持，但效应均低于原线，因此不据这组结果启动保持训练。数值、区间、有效域与独立复核由[报告](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)维护。

受测N是多阶段保留权重；已见TRAIN、IoU50与低输出阈值存在触顶条件。细边界、AP排序、新来源和更强变化仍未知。固定模型在本组变化下稳定，与更新模型后的能力保持是不同问题，夜间训练的保持效应按其自身条件解释。

[覆盖迁移](IDEA_PIG_COVERAGE_TRANSFER.md)和[受控合成](IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)继续保留开放范围，新的实际失效证据可改变后续选择。当前行动见[当前安排](../ACTIVE_PLAN.md)，执行记录见[Study](../../experiments/pig_appearance_stability_20261103/study.json)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_APPEARANCE_STABILITY_V1.md)保留数值、选择背景及反馈过程。
