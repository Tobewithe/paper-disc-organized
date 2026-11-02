---
schema_version: 1
id: IDEA_TRANSFERABLE_OCCLUSION_REPRESENTATION
kind: idea
title: 通过可控视频监督学习可迁移的遮挡与对应表征
summary: 用完整遮挡事件检验内容保持、可见证据变化和不确定性，关注迁移到实际对应与实例关联的作用。
outline_ids: [OUTLINE_VIDEO_OCCLUSION]
line_ids: [LINE_01701394050a46a3835bc63fb0d95f86]
exploration_status: unknown
assessment: unknown
author: AI 整理，依据既有主线说明
curation_status: source_based_draft
evidence:
  - path: project/knowledge_sources/occlusion_representation_pretraining_a946027d31d1996a.txt
    locator: 原文 paper/manuscripts/occlusion_representation_pretraining/README.md
    revision: a946027d31d1996a889123d1a88b3331d87a950eea2776222ef71cf1139a2b8c
    relation: references
---

## 想法与依据

仅能识别遮挡状态，不一定足以保持目标对应。由完整的可见、隐藏与重现过程获得监督，可能让表征同时保持内容身份并响应可见证据的变化。

## 怎样组织探索

[视频遮挡表征主线](../../paper/manuscripts/occlusion_representation_pretraining/README.md)已经提出视频生成、表征预训练与冻结迁移三个模块，并区分普通视频训练、历史信息和隐藏位置先验等对照。

生成监督、接口核验、表征可学习与真实任务获益分别需要证据。本条不从原入口中的作业快照判断当前终态，也不把既有视频身份实验算成本 Idea 的新结果。当前判断沿所属 Study 和后续记录核对。
