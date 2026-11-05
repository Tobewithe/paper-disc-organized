---
schema_version: 1
id: OUTLINE_PIG_HISTORY_EVIDENCE
kind: outline
title: 少量历史可见观察能否解除当前归属歧义？
parent_id: OUTLINE_PIG_OBSERVATION
outline_level: 3
summary: 判断少量真实看清片段增加了什么证据，并区分推理借用、训练对应、提示和时间平滑。
author: AI依据三层问题结构与既有文献整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: STARFORMER与猪视频基础模型系统中的输入、提示和额外帧
    relation: references
---

## 具体缺口与已有认识

当当前帧无法判定身体片段属于谁，少量过去看清的区域是否提供必要证据？现有视频系统有相关线索，但往往同时改变已知猪数、初始化、未来帧或质量控制；不能全归因于历史信息。

## 已排除范围与未知

旧视频／VOR研究按各自对象保留结论，尚不能替本项目猪实例任务裁决。原始序列是否可得、是否有可靠可见锚点，以及哪些当前错误能因此解除，仍未知。训练时借视频也不能令单帧恢复本帧不存在的信息。

## 关联Idea与可能方向

[历史可见证据Idea](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)可以探索推理时借用或训练时生成对应；与[可信归属](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)结合，历史失效时保留未知。实际探索结果应同时反馈到[观察问题](OUTLINE_PIG_OBSERVATION.md)与本页。
