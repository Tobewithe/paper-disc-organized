# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md](../../../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_SYNTHETIC_COUNTERFACTUALS
kind: idea
title: 用受控合成发现模型依赖与实例混淆的来源
summary: 将合成用于物理一致的成对变化与机制认识，为表示、监督和迁移产生想法，而不只扩充训练语料。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TASK, OUTLINE_PIG_DATA, OUTLINE_PIG_MODEL, OUTLINE_PIG_HYPOTHESES]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: paused
assessment: unknown
author: AI依据现有可控合成能力及用户开放探索要求推导
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 有限release的真实几何、标签资格及资产家族边界；不证明新成对场景已实现
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 合成猪检测、Copy-Paste与SAP显示内容、标签和迁移条件需分开考虑
    relation: references
---

## 想法与中心纲要的联系

同一头猪、同一场景中的哪些变化会让模型开始合并、漏分或错归？利用可控场景构造少量物理一致的变化：几何固定而改变照明／背景，个体外观保持而改变接触、遮挡或视角。改变物理因素后同步重渲图像与可见标签，观察哪些关系仍稳定、哪些开始失效。

它服务于[真实猪场实例分割这个大命题](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)的认识：区分问题来自观察、表示、场景依赖还是监督组织。所得知识可以导向真实标注、不同模型或额外观察，最终方法未必使用合成训练或部位标签。

## 为什么值得探索

既有工作已积累有限渲染与标签核验能力，过去主要把它们当训练辅助来源。另一种用途是把生成能力当作问题发现工具，以较小变化提高可解释性，而不以图像更多、更写实作为唯一进展。

例如，若遮挡几何固定而背景变化即导致归属改变，可能提示对场景外观依赖；若背景固定、只有可见断片关系变化时失效，则应认真考虑[片段表示](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)或[额外观察](../../../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)。这是生成竞争解释的方式，不是单次现象就能断定真实因果。

## 与其他方向的联系

可与[覆盖与迁移](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)连接，区分应改变的场景外观和应保留的几何信息；也可帮助研究[可见证据与歧义](../../../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)。可以比较通用模型，不依赖SPM、C/Q或某个原型接口。

## 未知和当前判断

[有限release](../../../../experiments/spm_concept_synthesis_repair_20261101/REPORT.md)不是本新对照已经实现的证据。单一身体／环境家族限制外推，合成敏感性也可能来自渲染伪影；必须保留真实适用性未知。反事实、受控渲染和鲁棒性压力分析有广泛前作，新颖性尚未核查。

## 已开展的首个认识实验（2026-11-03）

本Idea进入[条件可靠性主线](../../../../paper/manuscripts/pig_condition_reliability/README.md)，具体对象保存为[成像敏感性V1](../../../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)。[实际实验](../../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)完成128真实TRAIN色调变化及30组几何不变照明A/B：真实实例成功数损失2／1460、合成差异0／278，未支持本范围的大幅粗粒度能力损失。

本次展示了受控数据能用于约束一个具体解释；它没有证明背景依赖、所有光照变化无害、合成发现可外推或任何新算法有效。真实TRAIN和IoU50触顶、单一家族与温和照明范围均限制判断。工具方向暂缓后续执行、整体判断仍未知，其他干预可以在出现真实依据后形成新Idea，不自动扩量或开训。
