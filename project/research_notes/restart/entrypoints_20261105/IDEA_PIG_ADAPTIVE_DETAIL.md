# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_ADAPTIVE_DETAIL.md](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_ADAPTIVE_DETAIL
kind: idea
title: 为困难实例选择性获取细节与计算
summary: 根据对象冲突选择原图细节、局部重看或强模型，研究额外视觉证据的价值及全局身份一致性。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES]
line_ids: [LINE_PIG_ADAPTIVE_DETAIL_20261103]
exploration_status: paused
assessment: unknown
author: AI依据多尺度实例问题及用户开放探索要求推导
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构猪基准的细部差异、Mask2Former多尺度及基础模型系统的计算／输入条件
    relation: references
---

## 核心想法与大命题

全图预测后，真正需要更多细节的地方可能只是几处接触边缘、小肢体或对象冲突。探索如何选择这些区域，以原始高分辨率像素、局部重看或更强模型获取额外证据，并将结果保持在全局实例身份下。

它从视觉信息和计算分配切入[真实猪场实例分割](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，可以比较“所有帧增加计算”与“少数区域获取有价值信息”。不预定YOLO、固定ROI修补或某个基础模型是最终载体。

## 探索价值

如果全图缩放丢失了可辨细节，局部原图可能提供现有路径没有用到的信息；如果错误集中在少数对象，计算也可能更有针对性。与[关系询问](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)和[历史观察](../../../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)相连，可以进一步思考：一处歧义最值得用什么方式补信息。

这与旧固定候选池质量头的作用不同：后者主要选择已有输出，本方向可能改变实际观察与重新解析。旧质量头失败不能给它预判阳性，也不能作为全面否定。

## 关键未知

原始图像是否保留更多有效细节、当前错误是否由缩放造成、冲突选择是否可靠，尚未核实。只有低分辨率或原始模糊时，插值放大不会创造可见信息；完全遮挡也可能需要其他证据。局部处理还可能失去全局对象关系、产生重复实例，或损伤原本正确的区域。

自适应计算、粗到细分割、区域细化和专家调用都有前作，新颖性尚未核查。最初构思时仅保留为有条件潜力的候选，没有新图像或模型评价；后续局部病例资格及夜间训练阶段见下方实际反馈。原图是否有独立细节仍未知，若没有，应降低“新增高分辨率信息”这条具体途径的优先度。

## 2026-11-03建议：先验证本帧局部重看能否改善高IoU失败

结合[近期成像首试](../../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)在TRAIN的IoU50低输出阈值上触顶，以及[片段首试](../../../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)未能提供足够原标签断片，当前更值得直接检验真实精细Mask失败。尚未量化这类错误占比，不能先认定尺度或细节是主要瓶颈。

具体假说：对同一最终检测中框位置较准、MaskIoU介于0.50与0.75的实例，按预测框取得固定上下文局部并重新编码，可能利用本帧原有像素改善轮廓。现有640图即可做这个必要条件小试；它不等于取得独立高分辨率信息，也不恢复V25／V26固定池质量头。

建议最小范围：仅从既有128 TRAIN及原正常预测中，预先固定32个BoxIoU≥0.75且MaskIoU处于[0.50,0.75)的失败实例，并固定32个MaskIoU≥0.75成功对照。支持不足则如实收束该样本池，不为凑数自动扩量。使用同一原生模型与原GT口径，比较原全图、预测框周围固定上下文的局部重看，以及同裁剪／尺度但去除细节的对照；GT只用于诊断分组和评价，不能生成裁剪或挑选新预测。

暂定继续线：32个失败中至少8个达到MaskIoU≥0.75，较低细节对照至少多修复4个，32个原成功对照无新增失败。若两种裁剪同样改善，仅支持尺度／上下文的复合效应，不宣布细节机制；若无修复或伤害明显，停止当前方案。上述是建议的小试筛选线，尚未冻结，不能证明总体损伤≤1%或正常AP增益。

## 2026-11-03实际反馈与夜间新Study

用户授权实施后，[局部重看V1](../../../ideas/IDEA_PIG_LOCAL_RELOOK_V1.md)资格统计在原128 TRAIN只找到15个合格失败，未达32；1456个同条关联经服务器独立复核误差0。按原停止出口收束该池，方法未测；不把它当作广义细节或局部重看无效。

用户随后新增“一个晚上的较长训练序列，冒烟通过后排队”授权。[独立夜间训练Idea](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)现已全部完成：R/L/D/P Mask AP为52.734108／54.284877／51.143308／55.249199，均低于N57.343500。D未超过普通续训或低细节对照；P相对D提高4.105891点并降低损伤，但仍低于N且损伤4.4685%超过1%，全部当前配置结束。详见[结案报告](../../../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)。

本广义Idea暂缓／判断未知；本轮限制特定裁剪训练及保持配置，不否定新观察或所有细节信息。当前没有已获资格的训练后继。此Study使用全部原1272 TRAIN和固定432dev，是新授权训练问题，不是扩大原32例小试来挽救结果；全部实现与原始证据留服务器。

## 2026-11-04综合反馈：细节利用与保持作用分开

[夜间具体Idea的复盘](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)补清P−R=+2.515091、L−R=+1.550769，不能把本轮笼统描述成所有方法都不如普通续训。P仍低于多阶段保留N、损伤超限，D也低于同几何低细节L，因此继续停止这些具体配置；有限响应保持的相对作用值得保留为认识，不升级为可采用方法。

原局部推理资格不足、同帧缓存裁剪训练失败，以及独立高分辨率／历史观察的价值，是三个不同问题。[综合总结](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)没有增加后一个问题的直接证据，本广义Idea继续`paused/unknown`；未知与本版排除已回到[细节缺口](../../../outlines/OUTLINE_PIG_DETAIL_EVIDENCE.md)、[观察问题](../../../outlines/OUTLINE_PIG_OBSERVATION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

初步投入为零训练、约128次额外局部前向及有限工程复核；初步有效后仍需覆盖完整输出的AP、损伤和成本验证。以上为当时尚未执行的建议范围；实际资格停止和夜间新Study以本节反馈及各自协议为准。
