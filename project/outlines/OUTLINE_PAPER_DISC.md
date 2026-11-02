---
schema_version: 1
id: OUTLINE_PAPER_DISC
kind: outline
title: 项目知识总纲
summary: 从知识与问题出发，连接实例分割、视频遮挡和世界模型研究，追踪相关 Idea 的探索情况。
author: AI 整理，依据项目既有资料
curation_status: source_based_draft
evidence:
  - path: project/knowledge_sources/research_lines_dce4e168abcad55c.txt
    locator: 原文 project/research_lines.json
    revision: dce4e168abcad55c1efa54e8099af01b45590afe39d7d4af52bdccb679c61a05
    relation: references
  - path: project/knowledge_sources/idea_evolution_e93bf5358866376b.txt
    locator: 原文 project/research_notes/IDEA_EVOLUTION.md
    revision: e93bf5358866376b3bf31ffa649bf6fe01ba74e51a79d85866c8a9cdfaecc893
    relation: references
---

## 这份纲要怎样组织知识

本项目已有多个独立研究主线。这里先把相关知识、问题和未知组织起来，再关联从中形成的 Idea 与探索路径。纲要可以先于 Idea 存在；某个 Idea 或实验结束后，学到的认识仍应保留在知识体系中。

本稿是依据现有项目资料整理的初稿。关联的 Idea 条目用于重建知识与既有主线的联系，不作为最初立项时间、当前执行终态或科学结论的重新认定。

## 实例分割：已有信息怎样转化为正确实例

已有研究分别关注掩码读出、原型—系数表达，以及猪群场景中对象知识的获得和使用。共同需要分清：模型有无足够的信息、怎样读取和组织这些信息、修正错误时是否损伤已有正确结果。不同数据、分支和评价范围的结果各自保留，不能直接拼成统一成绩。

这一部分由子纲要[实例分割的信息、表示与读出](OUTLINE_INSTANCE_SEGMENTATION.md)展开。

## 视频遮挡：怎样利用观察历史保持对应

在遮挡发生、持续和解除的过程中，当前图像不一定提供充分证据。视频身份保持研究探索时序关联与恢复，遮挡表征预训练进一步研究如何获得可迁移的对应、可见性与不确定性表示。两条主线已有明确承接关系，既有实验和结论保留各自身份。

这一部分由子纲要[视频遮挡中的对应与身份保持](OUTLINE_VIDEO_OCCLUSION.md)展开。

## 世界模型：观察能否支持行动判断

世界模型主线关注时间采样和动力学可辨识性。它与视觉实例研究并列，讨论端点预测正确是否足以支持区间内切换动作的判断。这是独立研究主题，不继承其他主线的实验结果与约束。

这一部分由子纲要[世界模型的行动时间可辨识性](OUTLINE_WORLD_MODEL.md)展开。

## 让探索回到知识

每份子纲要记录已有认识、争议、未知及相关 Idea。研究主线组织探索，实验保留设计与证据，重要研究记录可以显式关联纲要与 Idea。阅读这些证据后再修订综述和判断，保留想法变化的依据。

当前执行安排仍查看[研究总控](../ACTIVE_PLAN.md)，研究身份查看[主线登记](../research_lines.json)。
