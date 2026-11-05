---
schema_version: 1
id: IDEA_PIG_OBSERVATION_INFORMATION
kind: idea
title: 少量看清过的观察，怎样帮助当前实例判断
summary: 利用可靠历史可见证据解决当前歧义，也探索视频提供训练对应；区分信息增加、平滑、提示和模型先验。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_TRUST, OUTLINE_PIG_HISTORY_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVALUATION, OUTLINE_VIDEO_OCCLUSION]
line_ids: []
exploration_status: candidate
assessment: unknown
author: AI依据用户澄清及猪视频系统文献作开放构思
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: STARFORMER与猪群Grounding-DINO/SAM2系统的输入、监督和质量控制条件
    relation: references
---

## 核心想法及联系

当前帧无法判断的肢体或身体断片，可能在少量历史帧中真实可见。探索怎样识别这些有价值观察、将它们与当前实例联系，以及何时历史已不可靠。目标是获得有根据的逐头分割，不预先规定采用长视频模型、固定记忆库或身份跟踪作为最终成果。

这与[中心纲要](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)的联系是改变可用信息。它可以独立于现有概念主线发展，也可为[片段归组](IDEA_PIG_INSTANCE_INFORMATION_USE.md)提供关系，或与[歧义表达](IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)结合，让判断保留证据来源。

## 两种值得思考的用途

- **推理时借用观察：**从过去看清的片段提供当前归属依据，研究取哪一帧、可信多久以及如何避免把旧错误反复传播。多视角或离线未来帧也是可能形式，其输入权限应单独说明。
- **训练时借用视频：**用可靠跨帧证据获得对应、局部伪Mask或弱关系，再学习单帧可预测的规律。它可能减少真实部位标签需求，但不会让单帧学生恢复本帧完全缺失的信息。

这些选择不需要先认定所有错误都是单帧不可辨，也不依赖暂停矩阵的结论才允许讨论。

## 价值、依据和未知

[已核猪视频文献](../research_notes/literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md)表明真实观察、强预训练与场景信息可以构成有用系统；同时，已知猪数、人工初始化、未来帧和质量控制会改变问题。STARFORMER中时间对比头的删除也未必损害分割，说明视频系统变好不等于某个时间loss有效。

潜力在于补充真实证据，而非只加强形状猜测。原视频可用性、可跟随的清楚锚点和本项目净收益仍未知；稀疏参考可能不能恢复长期完全遮挡，当前已有充分信息时也可能无需它。

本Idea与[视频遮挡知识](../outlines/OUTLINE_VIDEO_OCCLUSION.md)跨接，不将旧视频线或VOR的结果视为本猪分割假说已获验证，也不继承其运行、预算或提示权限。当前仅是候选。
