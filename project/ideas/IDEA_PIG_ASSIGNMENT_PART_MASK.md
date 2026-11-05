---
schema_version: 1
id: IDEA_PIG_ASSIGNMENT_PART_MASK
kind: idea
title: 分配—部位—Mask：实例条件概念监督的有限判别
summary: 原矩阵评价继续暂停，研究判断未知；训练末点与对账由所属Study和回执维护。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: paused
assessment: unknown
author: AI整理，依据原冻结协议及本地暂停记录
curation_status: source_based_draft
evidence:
  - path: experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json
    revision: 0b04a8d7d5006c2e030c5afeb8f719118b1c2730328c7297f2daabfc5d46576b
    locator: 2026-11-03只读核CP/QP各2000末点，原Run/SUMMARY四SHA及accounting；未恢复评价
    relation: references
  - path: project/research_notes/sources/assignment_bridge_protocol_eb8a1772f8e8708d.txt
    revision: eb8a1772f8e8708d2f2c3083938248bf67042592f63b6042fce99e36ca017f51
    locator: H_interface、H_assignment及首seed六臂有限矩阵
    relation: references
  - path: project/research_notes/sources/assignment_bridge_pause_7e86dc12ee5f9b0e.txt
    revision: 7e86dc12ee5f9b0e34995982c582e925bed004c8880982c4824879780ab065c7
    locator: 四臂完成、两臂未核、用户暂停；未作科学裁决
    relation: references
  - entity_id: spm_concept_assignment_bridge_20261102
    revision: 5a5977fe1c5fc6a0c8a1f66f5cef64d6bfec12776f99d6cbdfd495a6d033c137
    locator: 所属真实Study，原状态与预算独立保留
    relation: references
---

## 本版检验什么

检验部位监督绑定原生实例读出，以及Mask或部位质量参与训练分配，能否提供额外真实收益。C0/Q0、CM/QM、CP/QP用正确部位与同容量非解剖分区配对，区分语义、接口与分配作用。比较和停止条件由[冻结协议](../../experiments/spm_concept_assignment_bridge_20261102/PROTOCOL_V1.md)及[正式合同](../../experiments/spm_concept_assignment_bridge_20261102/FORMAL_PROTOCOL_V1.md)维护。

## 当前判断与未决项

评价继续暂停，正常效用与机制判断未知。[2026-11-03对账](../../experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json)补齐原CP/QP末点身份；训练完成和权重元数据不替代尚未执行的正常评价。实际Run及剩余事项见[Study](../../experiments/spm_concept_assignment_bridge_20261102/study.json)。

[既有TRAIN日志分析](../research_notes/PIG_NEXT_RESEARCH_PRIORITIES_20261102.md)显示分配干预实际改变很少，同时未支持辅助明显不可学或梯度幅度悬殊的解释。若后续结果阴性，需要结合实际改变的对象与剂量解释其范围；改变更多位置本身也未保证效用。

本版属于[猪体知识Idea](IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)的具体分支。接续从已核状态和原合同出发，暂停、test隔离与当前行动统一见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_ASSIGNMENT_PART_MASK.md)保留先前未核、后续对账及机制讨论的完整时点记录。
