---
schema_version: 1
id: OUTLINE_INSTANCE_SEGMENTATION
kind: outline
title: 实例分割的信息、表示与读出
parent_id: OUTLINE_PAPER_DISC
summary: 整理现有模型读出、对象知识与实例归属的联系，区分信息、学习和任务使用问题。
author: AI 整理，依据项目既有资料
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

## 要理解的问题

实例分割失败可能发生在目标提出、实例归组、局部边界读出、条件迁移和观察信息等不同环节。“分不好”本身不足以指定某个网络、原型库或部位监督方案。应先整理错误对象、可利用的信息和竞争解释，再形成可检验的 Idea。

## 已有表示的读出与修正

[原型—系数链路](../../paper/manuscripts/neighbor_sensitive_p3/README.md)研究固定模型中的可修机会、后训练兑现与损伤。[选择性掩码读出校准](../../paper/manuscripts/mask_boundary_calibration/README.md)研究一次修正是否有益，以及怎样减少对原成功对象的误伤。两条路径都需要区分可修上界与实际部署收益。

相关的[空间掩码读出稿件](../../paper/manuscripts/box_relative_mask_readout/README.md)也提供框相对位置、原生系数微调和空间读出的材料；它当前不在六条正式研究主线登记中，保留为知识来源。

## 对象知识怎样获得并用于实例任务

猪群分割中的思路从跨图像原型字典，发展到可变形对象、具名部位、内容参与重建，再到训练期对应和概念监督。[Idea 演变轨迹](../knowledge_sources/idea_evolution_e93bf5358866376b.txt)说明，这些变化并非每一步都证明前一步无效，而是在不断区分：知识可被表达、知识能被学习，以及任务是否实际使用这些知识。

SPM 主线是这些想法的既有探索路径。其目录名称保留历史身份，具体研究范围需要结合所属版本资料理解，不能让最初方法名称替代当前研究问题。

## 需要继续分辨的解释

[开放式文献分析](../knowledge_sources/pig_instance_analysis_e4e6e8d0e8969e75.txt)提出三个尚需区分的解释：实例表示或训练没有充分组织已有信息；相关条件覆盖与迁移不足；单帧观察本身缺少区分实例的信息。这三者可能共同作用，尚不能从综述本身确定主导原因。

## Idea 的探索情况怎样回到这里

本纲要关联现有读出修正、选择性校准和合成知识三个 Idea。具体方案、阴性、收益与限制沿原主线和实验查阅。新记录可显式关联本纲要，帮助更新对信息、读出和知识作用的认识；实验完成不自动等于某个机制成立。
