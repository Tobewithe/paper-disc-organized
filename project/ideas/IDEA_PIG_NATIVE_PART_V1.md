---
schema_version: 1
id: IDEA_PIG_NATIVE_PART_V1
kind: idea
title: 原生可见部位辅助V1及受限适配后继（已停止）
summary: 正确部位未取得稳定独立增量和满足保护的净效用；V1关闭，读出重放未支持受限适配V2。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
author: AI于2026-11-03据原正式报告及重放回填具体Idea身份；非新增实验或事前注册
curation_status: source_based_draft
scope:
  claim: 原V1全非BN适配下正确合成部位能相对Q提供稳定且满足正常效用／保护的额外收益。
  boundary: 原72训练／18QA辅助语料、2570参数接口、2000更新、三配对seed及预注册读出重放。
  retrospective_record: true
evidence:
  - path: experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md
    revision: 71fdc28133cda7a23fe99155240aea6d7343e9d5d3d059d4e06e5e0a9c6d4039
    locator: 四臂三seed正常评价、损伤及关闭V1和受限适配后继的依据
    relation: references
  - path: experiments/spm_concept_native_training_20261102/READOUT_REPLAY_V1.md
    revision: dba9401c912dd2ba6db943916902e69c8d339ca85012bd7a8f8f53391940be29
    locator: 预注册重放范围与受限适配V2的必要条件，未执行V2训练
    relation: references
---

## 本版主张与依据

从项目保留N适配原生非BN参数，比较真实续训R、几何辅助G、非解剖分区Q与正确部位C，检验正确语义能否提供独立真实收益并保护原成功实例。本条为原实验的事后Idea索引，执行身份沿原协议和Run保留。

## 当前判断与范围

V1关闭。C−Q小且种子异号，C低于保留N且损伤超限，稳定语义增量与可采用净效用的联合主张未成立。[正式报告](../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md)保存各臂数值和比较；[预注册读出重放](../../experiments/spm_concept_native_training_20261102/READOUT_REPLAY_V1.md)未满足受限适配V2的必要条件，V2未训练。

[训练审查](../research_notes/PIG_TRAINING_INTEGRATION_AUDIT_20261104.md)确认原生参数与辅助头实际更新、BN固定，降低了“只训练新头”或“BN漂移统一解释失败”的可信度。语义冗余、真实迁移不足与任务利用不足仍未区分。N是多阶段保留模型，R是匹配普通续训，语义增量继续由C−Q解释。

[同COCO起点版本](IDEA_PIG_COCO_JOINT_V1.md)已独立检验另一学习阶段；[分配矩阵](IDEA_PIG_ASSIGNMENT_PART_MASK.md)仍暂停。它们各自保留判断，广义[猪体知识](IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)仍未知。重新考虑部位方向需要新的信息或直接作用证据，已有辅助可学性不足以重开本版。

当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_NATIVE_PART_V1.md)保留原数值、后继排除及历史讨论。
