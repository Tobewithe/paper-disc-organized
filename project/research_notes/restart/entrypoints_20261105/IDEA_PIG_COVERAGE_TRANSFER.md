# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_COVERAGE_TRANSFER.md](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_COVERAGE_TRANSFER
kind: idea
title: 从同场景拟合走向新条件可靠性：覆盖、迁移与保持
summary: 区分可迁移的个体能力与可更新的场景信息，比较覆盖、少量真实监督和适配，不以结构标签价值为唯一目标。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_REAL_DATA, OUTLINE_PIG_DATA, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: paused
assessment: unknown
author: AI依据用户澄清及数据增强、迁移文献作开放构思
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: Copy-Paste、合成猪检测、SAP与DAFormer的迁移条件和限制
    relation: references
---

## 核心想法与中心命题

[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)面向真实猪场，不应把同一开发集的持续改进等同于新条件可靠性。研究相机、猪栏、光照、品种和生长阶段变化时，什么个体能力需要保持，什么场景信息可以少量更新。

一个值得展开的想法是：将可迁移的个体／实例能力与可更新的场景状态区分。几何、尺度与栏位可能是有用上下文，背景纹理或固定相机外观也可能形成偶然依赖；不是消除所有背景就更稳健。

## 可以打开的不同路径

有针对性的真实条件覆盖、低成本二维重组、合成变化、少量目标场景弱标、未标视频、教师学习和稳健适配，均是可能途径。选择应围绕缺失信息及新条件风险，不要求先有正确部位C对Q的增量。

可与[受控合成](../../../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)连接，分辨模型依赖场景外观还是关系几何；可与[关系标注](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)连接，思考新场景最值得补充哪类标注。数据、表示或训练都可能成为切入点。

一个更具体的可能性是利用固定相机：由空栏参考或长期录像形成设施、地面和活动区域地图，让模型借用长期稳定的场景结构，再适应照明或相机变化。这样既可能区分栏杆遮挡与动物轮廓，也提出“换一个场景时只需重建哪些上下文”的问题。它需要稳定视角及足够参考，当前尚未核实可得，也不意味着场景地图应直接重写原任务Mask。

## 现有线索及其边界

[90图release](../../../../experiments/spm_concept_synthesis_repair_20261101/REPORT.md)有有限标签资格，身体与环境各一个家族；它不证明真实覆盖充分。[概念V1](../../../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md)中真实单独适配也退化，提醒我们迁移与保持可能比新增某类语义更复杂，但没有识别唯一原因。

文献支持数据覆盖、监督强度与优化方式都可能改变迁移表现，尚未证明哪种是本项目瓶颈。当前只有有限来源，来源不自动等于独立猪场，反复使用432dev也不能建立跨农场结论。

## 从旧局部候选恢复开放范围

先前“C/Q×原覆盖／补缺覆盖”仅回答覆盖是否调节语义边际价值，是本Idea一个小问题，不定义整个迁移方向。不再规定本Idea只能作为相对归属候选的备选，也不把真实概念检查当所有覆盖、弱标或场景适配想法的前提。

已有Study的生成输入、GT、拆分和test隔离按原合同保持；2026-11-03用户新增首个验证授权，由独立的成像敏感性Study承接，不继承旧预算或恢复其他配置。

## 当前价值判断

跨场景可靠性具有长期价值，且能接纳多种低成本竞争方案。可用新来源、具体缺失条件和目标场景标签尚不明确，不立即收敛到某种域适配算法。

2026-11-03已完成[成像敏感性首试](../../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)：固定原生N，真实色调操作下的预定实例成功数损失仅0.136986%，30组合成照明配对差异为0，未达原效应线。因此不依据这组粗粒度损失进入保持训练。真实已见TRAIN、低输出阈值和IoU50触顶不能排除置信度、细边界或新场景风险，也没有检验几何／来源覆盖或适配方法。

## 2026-11-04综合反馈：成像鲁棒性、更新损伤与新域迁移分开

[综合复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)补入另一条限制：[夜间保持组P](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)比普通续训R高2.515091 AP，却仍低于项目保留N且原成功损伤4.4685%。这说明固定教师响应约束在本配置中缓解退化，但没有证明安全适配，更没有验证新相机／农场迁移。

[成像首试](../../../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)固定模型、本轮训练则更新模型，二者不能合并为同一种保持证据。概念／夜间的BN全冻也伴随退化，故BN不是已证统一解释；覆盖、标签、训练阶段和更新损伤仍需区分，本Idea继续`paused/unknown`，认识回到[迁移缺口](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)、[泛化问题](../../../outlines/OUTLINE_PIG_GENERALIZATION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

所属[条件可靠性主线](../../../../paper/manuscripts/pig_condition_reliability/README.md)暂缓新增执行，本广义Idea判断仍未知；具体V1假说已结束并限定排除范围。后续优先需要独立新条件或实际细粒度失败证据，而不是自动加大这组gamma或再加loss。
