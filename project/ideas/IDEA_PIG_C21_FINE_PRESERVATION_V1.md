---
schema_version: 1
id: IDEA_PIG_C21_FINE_PRESERVATION_V1
kind: idea
title: 限制C21细尺度适配并保持已学会的输出
summary: 局部更新较稳定，保持项已生效但边际作用很小；本版结束，保留原C21／V23。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRUST]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: mixed
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: C组末点、保持配对差、原成功损伤及GT75变化
    relation: references
---

## 本版主张与比较

[消融](IDEA_PIG_C21_V23_ABLATION_V1.md)表明细尺度承担修复并伴随局部损伤，[旧夜间保持](IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)则有相对作用。本版把更新范围限制到C21的fine_encoder与fine_refiner，比较相同预算的FINE与额外使用原C21输出KL的KEEP，检验保持约束能否改善修复／损伤取舍。

教师、其余参数及原策略固定，推理正常重算动作，具体监督区域与训练条件见[协议](../../experiments/pig_multi_idea_night_20261105/PROTOCOL_V1.md)。

## 当前判断与理由

本版结束，停止局部再适配及继续加KL的扩训。局部更新相对原C21较稳定，保持项实际生效，但KEEP与FINE的输出成功集合、损伤和净收益变化很小，额外教师前向增加训练成本。保留原C21／V23；详细配对、护栏及资源见[报告C组](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#6-c组局部更新较稳定kl没有带来实质增量)。

后继需要新的起点、参数范围或额外信息带来明确修复机会，并用匹配控制检验。如果输出、成功集合和损伤仍基本相同，应结束该保持对象。旧大范围保持的局部正效应继续按原起点和比较保存。

后继范围见[报告反馈节](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)，行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_C21_FINE_PRESERVATION_V1.md)保留原数值与复判过程。
