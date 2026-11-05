---
schema_version: 1
id: IDEA_PIG_COCO_JOINT_V1
kind: idea
title: 从共同COCO起点联合学习猪部位知识与实例分割
summary: 同COCO起点的正确部位未稳定超过非语义控制，本版关闭；一般辅助作用和共同长程退化分别保留。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: A组9个训练配置、三个预定点、配对控制与原成功损伤
    relation: references
---

## 本版主张与比较

从共同COCO预训练起点开始，在主要实例能力形成时加入正确部位，检验能否获得稳定语义增量。A_R为共同合成RGB曝光的无辅助控制，A_Q为非解剖分区，A_C为正确部位；C−Q检验语义增量，C−R检验辅助方案整体作用。选择这一比较的依据来自[基线与训练审查](../research_notes/PIG_TRAINING_INTEGRATION_AUDIT_20261104.md)，具体训练条件见[协议](../../experiments/pig_multi_idea_night_20261105/PROTOCOL_V1.md)。

## 当前判断与理由

本版结束并停止扩训。正确部位未稳定超过Q，三个预定学习点的配对均差均未转正；C与Q相对R的改善更支持一般辅助作用。R／Q／C随训练共同退化，使普通适配过程成为独立未决问题。数值、学习曲线与收益损伤由[报告A组](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#4-a组一般辅助有作用换起点仍未获得语义特异增量)维护。

新版本需要有依据的训练／信息瓶颈或能够区分语义与普通辅助的新信息，再用匹配控制检验。已有辅助loss下降、非零梯度和起点变化已被本轮考虑，继续投入需超出这些依据。广义[猪体知识Idea](IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)仍未知。

汇总诊断及后继范围见[报告反馈节](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)，实际行动统一见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_COCO_JOINT_V1.md)保留详细数值和历次复判。
