# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md](../../../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_FIXED_POOL_QUALITY_V25_V26
kind: idea
title: 固定V23候选池内的质量头扩展（V25／V26，已停止）
summary: 保存两个已测质量学习配置的取舍；均低于V23且损伤超限，停止该固定池质量头扩展，不外推为所有选择或细化方法无效。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_TRUST, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
author: AI于2026-11-03据原完整报告回填具体Idea身份；非新增实验或事前注册
curation_status: source_based_draft
scope:
  claim: 在原固定候选池与预算内，V25／V26质量学习能够超过V23并通过损伤保护。
  boundary: 仅已测固定池、输入、目标、实现与训练预算。
  retrospective_record: true
evidence:
  - path: experiments/spm_action_quality_20261026/REPORT.md
    revision: 3d6163669ea048785a3ad15742b3dce30e59d92657d1b238c2b5f9e37f68ad7d
    locator: V25完整实验、正常结果、原损伤护栏及停止扩训决定
    relation: references
  - path: experiments/spm_native_spatial_quality_20261026/REPORT.md
    revision: eb912593c3945fd2011c92dec0d4905456daa7c384678bfe496ac657b3cc7fe3
    locator: V26增加冻结原生局部特征后的结果及停止固定候选质量头扩展决定
    relation: references
---

## 当时想验证什么

有限候选池有GT辅助机会，是否能够借正常输入的质量学习兑现？V25读取候选Mask、RGB和对象图，V26进一步增加冻结原生P2／P3局部特征；候选、动作与成本仍属原固定范围。这里的具体主张是得到超过V23且通过保护的正常收益，不是“只要高于原生就成功”。

这是据原报告补建的具体想法记录，主张、预算和Gate仍以原协议为准；原报告保存各实际Run与产物来源，不将今天的稳定ID冒充当时预注册。

## 实际结果与停止理由

[V25原报告](../../../../experiments/spm_action_quality_20261026/REPORT.md)：Mask AP60.272390，较V23低0.480889点，原成功损伤37/2126＝1.740357%。

[V26原报告](../../../../experiments/spm_native_spatial_quality_20261026/REPORT.md)：Mask AP60.000682，较V23低0.752596点，损伤33/2126＝1.552211%。两个配置均未过原1%护栏；V26之后原报告明确停止固定候选质量头扩展。它们相对原生有收益，仍不能满足这个具体主张。

当前记录为“已关闭、该范围内主张被结果反驳”，保留两版证据及失败，不继续在同池内靠换头、加输入或换loss追结果。GT池有好答案也不等于正常输入一定可学会选择。

## 排除范围与保留问题

停止的是这组固定候选质量学习扩展，不是所有候选选择、局部观察或掩码学习。是否有其他可预测的纠错信息仍开放；若未来重新考虑，需说明旧失败中哪个前提发生了有依据的变化，仅重述oracle机会不足以重开。

[按需原图细节](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)可能改变实际观察，[可见片段组织](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)可能改变实例表示，它们是不同问题，尚未由本实验验证。潜在联系不能作为新方案已获支持的证据，也不自动恢复旧后置路线。

## 2026-11-04已有选择策略的功能作用

[C21／V23固定权重消融](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)中，top1全强度应用／完整候选直接应用分别下降4.777370／3.750892 AP，原N成功损伤由18增至244／178。现有选择策略在当前模型中保护已有正确预测，保留原V23选择路径的依据增强。本Idea所测的“继续扩展质量头以超过V23”仍按V25／V26原结果关闭；后续研究优先分析内容与选择的分工。

## 回到纲要

结果已沉淀到[中心纲要的已有取舍](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)；未来候选及其未知见[开放探索](../../../outlines/OUTLINE_PIG_HYPOTHESES.md)。具体执行只沿原总控，不由本条重新授权。
