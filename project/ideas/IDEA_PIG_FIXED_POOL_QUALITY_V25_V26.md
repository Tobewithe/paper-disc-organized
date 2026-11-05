---
schema_version: 1
id: IDEA_PIG_FIXED_POOL_QUALITY_V25_V26
kind: idea
title: 固定V23候选池内的质量头扩展（V25／V26，已停止）
summary: 两个固定池质量学习版本低于V23且损伤超限，停止该扩展；其他观察、表示和可预测纠错信息仍开放。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_TRUST, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
author: AI于2026-11-03据原完整报告回填具体Idea身份；非新增实验或事前注册
curation_status: source_based_draft
scope:
  claim: 在原固定候选池与预算内，V25／V26质量学习能够超过V23并通过损伤保护。
  boundary: 仅已测固定池、输入、目标、实现与训练预算。
  retrospective_record: true
evidence:
  - path: experiments/spm_action_quality_20261026/REPORT.md
    revision: 3d6163669ea048785a3ad15742b3dce30e59d92657d1b238c2b5f9e37f68ad7d
    locator: V25完整实验、正常结果、原损伤护栏及停止扩训决定
    relation: references
  - path: experiments/spm_native_spatial_quality_20261026/REPORT.md
    revision: eb912593c3945fd2011c92dec0d4905456daa7c384678bfe496ac657b3cc7fe3
    locator: V26增加冻结原生局部特征后的结果及停止固定候选质量头扩展决定
    relation: references
---

## 本版主张

在原V23固定候选池和预算内，质量学习能否兑现GT辅助机会，并取得超过V23且满足保护的正常收益？V25读取候选Mask、RGB和对象图，V26再增加冻结原生局部特征。当前Idea为原实验的事后索引。

## 当前判断与依据

两个版本均低于V23、损伤超限，停止固定池质量头扩展。具体结果和原停止依据见[V25报告](../../experiments/spm_action_quality_20261026/REPORT.md)与[V26报告](../../experiments/spm_native_spatial_quality_20261026/REPORT.md)。

后续[固定权重消融](IDEA_PIG_C21_V23_ABLATION_V1.md)支持已有选择策略的保护作用，因此保留原V23路径；这与继续扩展质量头能否超过V23是不同主张。

## 保留问题与复判依据

其他可预测纠错信息、[新的局部观察](IDEA_PIG_ADAPTIVE_DETAIL.md)或[实例表示](IDEA_PIG_INSTANCE_INFORMATION_USE.md)仍有开放空间。重新考虑时需要指出旧失败中的哪个前提发生了有依据的变化；原池存在好答案这一事实已经纳入本次判断。

当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)保留原数值、消融反馈与排除范围。
