# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md](../../../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_ASSIGNMENT_PART_MASK
kind: idea
title: 分配—部位—Mask：实例条件概念监督的有限判别
summary: 已有YOLO六臂各2000末点元数据已核；原矩阵评价仍暂停，无dev效用或机制结论。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: paused
assessment: unknown
author: AI整理，依据原冻结协议及本地暂停记录
curation_status: source_based_draft
evidence:
  - path: experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json
    revision: 0b04a8d7d5006c2e030c5afeb8f719118b1c2730328c7297f2daabfc5d46576b
    locator: 2026-11-03只读核CP/QP各2000末点，原Run/SUMMARY四SHA及accounting；未恢复评价
    relation: references
  - path: project/research_notes/sources/assignment_bridge_protocol_eb8a1772f8e8708d.txt
    revision: eb8a1772f8e8708d2f2c3083938248bf67042592f63b6042fce99e36ca017f51
    locator: H_interface、H_assignment及首seed六臂有限矩阵
    relation: references
  - path: project/research_notes/sources/assignment_bridge_pause_7e86dc12ee5f9b0e.txt
    revision: 7e86dc12ee5f9b0e34995982c582e925bed004c8880982c4824879780ab065c7
    locator: 四臂完成、两臂未核、用户暂停；未作科学裁决
    relation: references
  - entity_id: spm_concept_assignment_bridge_20261102
    revision: 5a5977fe1c5fc6a0c8a1f66f5cef64d6bfec12776f99d6cbdfd495a6d033c137
    locator: 所属真实Study，原状态与预算独立保留
    relation: references
---

## 这个版本实际检验什么

在原概念Goal下，以已有YOLO为受测模型，检验把部位监督绑定到原生实例读出，以及普通Mask质量或部位质量参与训练分配，是否带来额外真实收益。正确部位与同容量非解剖分区配对；不把“额外实例处理”自动归为猪解剖知识。

六臂为C0/Q0、CM/QM、CP/QP。首seed按原合同各2000更新，先评价再决定后续seed。具体数值、控制与停止条件见[冻结协议](../../../../experiments/spm_concept_assignment_bridge_20261102/PROTOCOL_V1.md)及[正式合同](../../../../experiments/spm_concept_assignment_bridge_20261102/FORMAL_PROTOCOL_V1.md)，本条不重定门槛。

## 本地已核的进度

工程资格及六臂首seed训练已核。1146/1148/1150/1152对应四臂各2000更新；2026-11-03服务器恢复后，1154/1156的Slurm均COMPLETED、0:0，CP/QP原Run和SUMMARY也确认2000末点，四文件原SHA核验回传。见[最新对账](../../../../experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json)。末点权重仍留服务器，未据摘要SHA冒充本地完整权重复算。

原暂停记录保留其当时两臂未核事实；本次新线指令支持上述只读核验，不自动恢复旧矩阵评价或后两seed。当前无本矩阵的dev效用或机制结论，test426仍封存。进度继续为“已暂停”，判断“未知”，不写科学阴性。

## 新增的机制解释边界（2026-11-02）

根据已回传TRAIN日志的[核读及来源](../../PIG_NEXT_RESEARCH_PRIORITIES_20261102.md)，CM/QM真实一对一分支累计raw状态变化只有4/2，对应累计原正样本92017；它们是训练事件计数，不是独立猪数或直接替换比例。Q loss下降，指定原型卷积层五时点Q/C加权辅助梯度比中位约1.014，暂不支持明显不可学/幅度悬殊解释。

这些事实尚不能判方法阳性或阴性，也不使原协议失效。它们要求把池内可用信息、实际干预和正常效用分别解释：若M不改善，首先约束该实现，不能充分排除分配机制；旧[多位置监督](../../../../experiments/mask_supervision_support_20261004/REPORT.md)又表明改变大量位置不保证有用。

[论文叙事中的候选C](../../../../paper/manuscripts/spm_yoloseg/PAPER_NARRATIVE_AND_IDEAS_20261102.md)将其作为先行诊断，不宣称新颖算法；只复用已有量，缺量时另列有限TRAIN核验设计，不全raw重扫、不改Gate、不因剂量小自动调强重训。

## 与开放分析及历史Idea的关系

它属于[猪体知识探索](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)的一个具体分支。该版本选用YOLO，不将其升级为整个[猪实例分割知识纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)的默认模型；新竞争性Idea也不继承本Study的授权或预算。

本条保留原冻结实验身份与数值门槛。两作业只读对账已完成；本次实际执行转入独立片段关系主线，没有恢复本矩阵评价或重提其训练。若以后接续，应从上述已核事实出发，按所属协议明确剩余动作，不重复恢复已完成的两臂。
