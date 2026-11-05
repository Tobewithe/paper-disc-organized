---
schema_version: 1
id: OUTLINE_VIDEO_OCCLUSION
kind: outline
title: 视频遮挡中的对应与身份保持
parent_id: OUTLINE_PAPER_DISC
summary: 整理时序证据、可见性、对应和不确定性，连接身份保持与可迁移视频表征的探索。
author: AI 整理，依据项目既有资料
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/dense_video_identity_37f6d402c8c7c371.txt
    locator: 原文 paper/manuscripts/dense_video_identity/README.md
    revision: 37f6d402c8c7c3712dd3905769b4acb4e6391f4347e57ba9c6ef767e8053d24e
    relation: references
  - path: project/research_notes/sources/occlusion_representation_pretraining_a946027d31d1996a.txt
    locator: 原文 paper/manuscripts/occlusion_representation_pretraining/README.md
    revision: a946027d31d1996a889123d1a88b3331d87a950eea2776222ef71cf1139a2b8c
    relation: references
---

## 遮挡改变了什么

受到遮挡、离开视野、观测模糊和身份发生混淆是不同情形。当前帧没有充分证据时，历史观察可能帮助恢复对应；完全不可辨的交换仍应保留不确定性，不能把连续轨迹输出本身当作正确恢复。

## 从身份保持到表征学习

[密集视频身份主线](../../paper/manuscripts/dense_video_identity/README.md)已经积累关联、确认、回退、可见性学习及真实视频评价材料。源记录保留局部收益、相对强基线的差距与机制尚未成立的结果。

[视频遮挡表征预训练](../../paper/manuscripts/occlusion_representation_pretraining/README.md)明确承接后续核心方法研究：关注完整遮挡事件、可控监督、内容保持、可见证据变化，以及同一个表征在对应和实例关联任务中的迁移。

## 已有知识与开放问题

三维场景能够提供可复核的几何与对应监督，这是研究依据；监督能够生成、合成状态能够学习与真实任务稳定获益，需要分别验证。研究还需区分普通视频训练、中间历史、隐藏位置先验和新增约束各自贡献。

## 相关 Idea 与证据

本纲要分别关联“利用时序证据保持身份”和“学习可迁移的遮挡对应表征”两个 Idea。前者为后者提供已有证据与下游应用背景，旧 Run 不转换为新主线结果。具体进展沿所属 Study 与研究记录查阅，本稿不确认远端任务当前终态。
