---
schema_version: 1
id: OUTLINE_PIG_DETAIL_EVIDENCE
kind: outline
title: 原图细节是否提供全图路径没有利用的信息？
parent_id: OUTLINE_PIG_OBSERVATION
outline_level: 3
summary: 区分有用的高分辨率观察与纯插值，研究局部重看怎样保持全局实例身份及合理成本。
author: AI依据三层问题结构与既有研究整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构猪细部差异、多尺度表示与基础模型输入条件
    relation: references
---

## 具体缺口与已有认识

接触边缘、小肢体或细部错误，是否来自全图缩放丢失了原图可辨信息？如果存在，局部原图重看可能提供已有路径没有利用的证据；如果原始图像已模糊或区域不可见，放大不必有帮助。

## 已排除范围与未知

[固定候选质量头V25／V26](../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)停止的是既定候选评分扩展，不等于已检验更高分辨率的新观察。当前原图额外细节、错误类型及选择可靠性尚未核。局部处理还可能丢失对象全局关系，产生重复或损伤。

## 2026-11-03的实际反馈

[局部重看最小验证](../../experiments/pig_local_relook_20261103/REPORT_V1.md)在原128TRAIN的1456个同条框75关联中，只找到15个MaskIoU0.50–0.75失败，低于预定32；独立关联及位运算重算误差0。原入口支持不足、方法未测，没有通过扩图或改阈值挽救。这个结果限制本次抽样入口，不能证明更高分辨率信息或局部计算无价值。

用户随后要求较长夜间训练序列，另立[真实细节训练Idea](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)：R完整图、L同几何低细节、D细节局部、P原响应保持。现已全部完成，各16500步；D低于R和L，P较D提高4.105891 AP并降低损伤，但P仍低于N2.094301点、损伤4.4685%超1%，所有继续线失败，当前配置结束。见[完整报告](../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)。这轮不支持特定局部细节训练的净收益，不能据此说所有新细节观察无价值；保持项有相对作用也不等于可采用。

## 2026-11-04综合认识回流

[夜间具体Idea](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)补充了P−R=+2.515091、L−R=+1.550769：同条件普通续训和多阶段保留N是不同参照，不能用低于N说所有训练方法都无相对作用。P的保持作用尚未跨过采用与损伤线；D−L为负也只限制本版低通／细节构造。

[综合复盘](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)把三件事继续分开：原128TRAIN的局部推理资格不足；同640缓存上的裁剪训练未获净效用；独立原图细节或历史观察价值未测。[广义细节Idea](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)仍暂缓／未知，本缺口保持开放并回到[观察问题](OUTLINE_PIG_OBSERVATION.md)，没有新的训练后继。

## 关联Idea与可能方向

[按需获取细节](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)可与[片段组织](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[关系询问](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)比较信息价值。探索前先明确是否真的增加像素证据，结果反馈到[观察问题](OUTLINE_PIG_OBSERVATION.md)，不自动重开旧修补路线。
