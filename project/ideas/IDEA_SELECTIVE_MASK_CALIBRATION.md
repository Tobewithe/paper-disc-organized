---
schema_version: 1
id: IDEA_SELECTIVE_MASK_CALIBRATION
kind: idea
title: 预测修正收益，选择性校准掩码读出
summary: 用可观测的预测信息判断一次修正是否有益，减少统一收紧掩码造成的误伤。
outline_ids: [OUTLINE_INSTANCE_SEGMENTATION]
line_ids: [LINE_f2b78b79b389409f80ed61f8480ee459]
exploration_status: unknown
assessment: unknown
author: AI 整理，依据既有主线说明
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/mask_boundary_calibration_6c4f357184adf2d3.txt
    locator: 原文 paper/manuscripts/mask_boundary_calibration/README.md
    revision: 6c4f357184adf2d3adc4b6869e301c90275722c9da49866068702a95690224b8
    relation: references
---

## 想法与依据

提高掩码阈值可以修复部分多余前景，也可能删除正确目标像素。由此产生选择性修正的想法：先预测一次操作的收益，再决定是否采用。

## 已有探索怎样理解

[研究主线](../../paper/manuscripts/mask_boundary_calibration/README.md)包含 RCMC 历史稿件、冻结协议结果和后续机制探索。论文稿件交付与后续探索是否完成需要分别理解，不混合不同评价口径。

本条连接已有想法与知识纲要；具体收益、损伤及适用边界沿原证据查看。当前主线进度和整体研究判断在核对后更新。
