---
schema_version: 1
id: IDEA_PIG_KNOWLEDGE_FOR_INSTANCES
kind: idea
title: 获得并利用猪体知识，改善接触与遮挡中的实例归属
summary: 追踪原型、对象与部位知识怎样被学习并实际用于实例分割，保留竞争解释与思路演变。
outline_ids: [OUTLINE_INSTANCE_SEGMENTATION]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: unknown
assessment: unknown
author: AI 整理，依据项目 Idea 演变与研究资料
curation_status: source_based_draft
evidence:
  - path: project/knowledge_sources/idea_evolution_e93bf5358866376b.txt
    locator: 原文 project/research_notes/IDEA_EVOLUTION.md
    revision: e93bf5358866376b3bf31ffa649bf6fe01ba74e51a79d85866c8a9cdfaecc893
    relation: references
  - path: project/knowledge_sources/pig_instance_analysis_e4e6e8d0e8969e75.txt
    locator: 原文 literature_search/pig-instance-open-analysis_2026-11-02/summary.md
    revision: e4e6e8d0e8969e7504bd9ad4bfc671b14f66255ec0987adaebc8aadd9c51de88
    relation: references
---

## 想法与知识问题

怎样让模型获得并利用关于猪的知识，从而区分接触、遮挡中的不同实例？知识保存在哪里、怎样学到、怎样用于任务，是需要分别回答的问题。

## 已有探索与演变

[主要 Idea 演变轨迹](../knowledge_sources/idea_evolution_e93bf5358866376b.txt)记录了视觉原型字典、身体结构、具名部位记忆、内容参与重建、共同坐标、训练期对应和概念监督等变化。这些分支与回转保留原有依据，不能将它们压成一条所有前案都被否定的直线。

[SPM 研究主线](../../paper/manuscripts/spm_yoloseg/README.md)继续作为已有探索入口。该条目只连接其知识问题，不把旧目录名称固定为未来方法选择。

## 仍需辨别的解释

[开放分析](../knowledge_sources/pig_instance_analysis_e4e6e8d0e8969e75.txt)把实例关系利用、条件覆盖与迁移、观察信息不足列为竞争解释。概念可预测与实例任务得到额外收益需要不同证据。具体版本的结论和暂停安排沿原实验与总控核对，此处暂不合并成整体裁决。
