# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md](../../../ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_FRAGMENT_RELATION_PILOT
kind: idea
title: 当前真实TRAIN能否支撑可见片段关系研究？
summary: 完成真实TRAIN适用性实验：可靠原断连仅14实例／14图，未达最低支持量；停止用当前缓存直接训练显式断片关系。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_REAL_DATA]
line_ids: [LINE_PIG_FRAGMENT_RELATION_20261103]
exploration_status: closed
assessment: insufficient
author: AI依据用户开展实验指令及当前数据身份形成的具体首试
curation_status: source_based_draft
scope:
  claim: 既有真实TRAIN原独立Mask中存在足够可追溯的非平凡同体断片，可作为后续关系学习研究对象。
  current_stage: s0_completed_current_cache_entry_closed_insufficient_support
  exclusions: 非新增标注成本结论，非物理可见真值，非模型或机制效用。
evidence:
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 1272图／14642实例，可靠原断连14实例／14图／3个声明group；预定支持量未达到
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: CClusnet提供直接猪领域片段问题依据，但不代替目标数据分母
    relation: references
---

## 从哪个缺口形成

[片段表示Idea](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)和[关系标注Idea](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)值得结合，但当前尚不知道真实监督里是否有足够可追溯片段。现有原独立Mask有重叠，160网格也可能改变细部连通性；因此不能直接把不连通叫作物理遮挡。

## 具体首试

在固定1272 TRAIN、14642实例缓存上统计原Mask的非平凡连通分量，并检查每个分量在原独立标签中的唯一归属支持。另报剔除重叠后诱发的断连，不把它替换成主对象。固定算法、阈值、全量分母、样例选择和停止出口见[冻结协议](../../../../experiments/pig_fragment_relation_20261103/PROTOCOL_S0.md)。

若可靠原断片少于128实例、32图或2个声明group，不进入本数据上的片段方法训练；这是最低工程支持量，不是统计显著性或任务失败率阈值。若通过，只支持继续核视觉语义和模型错误，不宣布关系学习有效。

## 实际结果与取舍（2026-11-03）

Slurm1157真实完成，16项产物全SHA核验回传；完整结果见[实验报告](../../../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。14642实例的原Mask非平凡断连15例，可靠支持14例／14图／3个声明group，占全部实例0.095615%，未达预定128例／32图。另有150例只在剔除重叠后才形成断连，不改作主对象。

因此结束当前缓存的显式断片学习入口，进度“已结束”、判断“支持不足”。这不代表关系学习被方法实验否定，也不能由原标注断片少推断真实遮挡或归属错误少。[广义信息利用Idea](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)保持判断未知，[关系标注Idea](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)尚未测人工效率。

## 2026-11-04综合回顾的边界

[近期总结](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#6-三个首试没有方法成绩的地方保留未测)复核本条仍为`closed/insufficient`：只排除当前160缓存直接提供足够断片支持的入口，没有关系模型效用结果。它不能与三组训练阴性合并成“所有信息方向都无效”；[片段缺口](../../../outlines/OUTLINE_PIG_MODEL.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)继续保留明确可见标注、序列或真实归属错误可能提供新依据的未知，本次未新增观测。

## 状态与追溯

所属[片段关系主线](../../../../paper/manuscripts/pig_fragment_relation/README.md)暂停于输入支持不足，[Study](../../../../experiments/pig_fragment_relation_20261103/study.json)保存真实Run、30 CPU-core秒／0 GPU秒及产物。冻结的原Idea版本保留在实验`frozen/`，当前状态没有追改启动前主张。结果已回到[具体缺口](../../../outlines/OUTLINE_PIG_MODEL.md)和[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

高分辨率原标注、明确可见监督、可核序列或实际归属错误可能提供新的研究依据，尚未检验。重新推进应先说明新增证据改变了哪一前提，不通过调阈值、制造断片或借旧SPM预算挽救本版。
