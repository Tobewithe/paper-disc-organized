---
schema_version: 1
id: IDEA_PIG_ADAPTIVE_DETAIL
kind: idea
title: 为困难实例选择性获取细节与计算
summary: 探索困难实例的额外视觉证据与计算分配；当前病例池和裁剪训练入口已收束，独立细节的价值仍未知。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES]
line_ids: [LINE_PIG_ADAPTIVE_DETAIL_20261103]
exploration_status: paused
assessment: unknown
author: AI依据多尺度实例问题及用户开放探索要求推导
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构猪基准的细部差异、Mask2Former多尺度及基础模型系统的计算／输入条件
    relation: references
---

## 想法与价值

对少数接触边缘、小肢体或对象冲突，选择原图细节、局部重看或更强模型获取额外证据，并保持全局实例身份。它从视觉信息和计算分配切入[中心命题](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，可与[关系询问](IDEA_PIG_RELATION_ANNOTATION.md)、[历史观察](IDEA_PIG_OBSERVATION_INFORMATION.md)比较补充信息的价值。

如果全图缩放丢失可辨细节，局部观察可能有帮助；若原图模糊或完全遮挡，则需要其他信息。区域选择、全局关系保持和额外计算成本也影响效用。具体模型与输入尚可调整，区域细化和自适应计算已有前作。

## 当前判断与证据

广义Idea暂缓、判断未知。已有两项具体检验分别收束：

- [局部重看V1](IDEA_PIG_LOCAL_RELOOK_V1.md)在原病例池中支持不足，局部推理方法未测。
- [独立夜间训练](IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)未支持本版裁剪／细节配置的净效用。保持项相对普通续训及无保持有作用，但仍低于多阶段保留N且损伤超限，当前配置结束。

原局部推理资格、同帧裁剪训练与独立高分辨率信息是不同问题。原图是否提供新增有效细节、实际错误是否由缩放造成、什么选择策略能控制损伤仍未解决；新的观察或明确失效证据可以改变选择。

探索由[细节主线](../../paper/manuscripts/pig_adaptive_detail/README.md)组织，当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_ADAPTIVE_DETAIL.md)保留原小试建议、后续训练与复盘，不再在本入口重复维护各协议预算和结果表。
