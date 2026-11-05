---
schema_version: 1
id: IDEA_PIG_C21_V23_ABLATION_V1
kind: idea
title: C21与V23的收益依赖哪些实际组件？
summary: 固定权重消融支持内容、细尺度、外观和选择保护的作用；形变与逐部位保留进入简化候选，正确对应的增量仍开放。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: mixed
author: AI依据用户2026-11-04要求先做C21和V23消融形成，并按原实验及验收结果更新
curation_status: source_based_draft
evidence:
  - path: experiments/pig_c21_v23_ablation_20261104/RESULT_SUMMARY.json
    revision: 652dc5f6fa03f46f73ef0b45ad735020b0ee89b1d2f87358e8024e9512e43e31
    locator: 完整432dev、十组正常输出及七项功能干预
    relation: references
  - path: experiments/pig_c21_v23_ablation_20261104/ACCEPTANCE_RECEIPT.json
    revision: 042d65c77d9ffa3f5cc32a02389360ec9fb2e8ce4c68c83c9271e70eed8fa618
    locator: 1176独立完整性、COCO指标、GT状态及来源重抽复核
    relation: references
---

## 本次检验

对固定C21／V23，在正常推理路径移除内容、细尺度、外观、形变、逐部位保留或选择保护，并允许动作／风险重新计算，辨别收益依赖哪些组件。它与旧固定动作和修正量的干预条件不同，比较对象见[协议](../../experiments/pig_c21_v23_ablation_20261104/PROTOCOL_V1.md)。

## 判断及研究影响

本轮已结束，判断为混合。内容、细尺度、外观移除明显降低性能；强制完整应用增加修复，同时带来更大的损伤和空Mask，支持选择保护在当前模型中的作用。形变置零和平均共享保留的影响较小，使它们成为简化候选。完整数值、正常对照与独立验收见[报告](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)。

据此保留原V23，优先理解内容重建、外观匹配和选择保护的配合。[功能简化重训](IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)已继续检验低依赖组件的约束；保持内容总量时正确空间／部位对应的额外作用仍待辨别。该问题更新[广义猪体知识Idea](IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)，不把置零效应直接归为正确对应的贡献。

当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_C21_V23_ABLATION_V1.md)保留原干预表、作业与反馈过程。
