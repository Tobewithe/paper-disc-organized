# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md](../../../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_APPEARANCE_STABILITY_V1
kind: idea
title: 固定几何时，成像外观变化是否破坏实例区分？
summary: 128真实TRAIN的预定最坏实例成功数损失仅2/1460，30组合成A/B差异0/278；未达到大幅损失预测，结束本范围验证入口。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TRANSFER_GAP, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_DATA, OUTLINE_PIG_TASK, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: closed
assessment: contradicted
author: AI依据用户从剩余缺口选择并完成首个验证实验的指令形成
curation_status: source_based_draft
scope:
  claim: 当前原生模型在几何和标签不变的成像变化下，会丢失足够多可同时区分的实例，值得继续研究条件保持。
  stage: completed_large_effect_below_frozen_size
  tested_scope: 协议V1的真实损失至少5%及合成差异至少10%预测，限定低阈值正常输出的160格点IoU50计数。
  limits: 真实TRAIN数字色调与有限合成照明分别解释；非方法效用、非独立泛化和背景捷径证明。
evidence:
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 完整冻结样本、真实0.136986%及合成0的主量、区间和未达效应线判断
    relation: references
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 30个几何组的A/B灯光对照及辅助标签资格，身体／环境各一家族
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 跨条件、覆盖和域适配提供相关问题依据，不代表本假说已成立
    relation: references
---

## 缺口、价值与选择

[新条件下的能力保持](../../../outlines/OUTLINE_PIG_GENERALIZATION.md)需要先区分数据覆盖、成像依赖和更新损伤。[受控合成Idea](../../../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)提供改变条件的工具，[覆盖迁移Idea](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)提出为什么这些差异值得研究。本具体首试检验冻结模型的反应，当前不预定保持方法。

前一[片段缓存首试](../../../ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)已关闭，不通过重切其Mask恢复。剩余历史观察缺可核序列，按需细节缺独立原图信息，关系标注缺真实成本；成像变化有现成的几何不变对照，因此优先做一次能直接改变判断的有限实验。详细比较见[选择依据](../../../../experiments/pig_appearance_stability_20261103/DESIGN_AND_SELECTION.md)。

## 区别预测与首个实验

固定项目保留YOLO权重N、正常无GT前向和解码。在128张按源ID哈希预选TRAIN上做原图／bright／dark，在全部30组A/B合成上核几何与标签一致后比较。真实主读数是两种预定变化造成的每图实例成功数最坏损失；合成只在原owner有效域内做诊断，标签权限不变。

真实与合成都敏感，才共同支持进一步研究成像保持；仅合成敏感不能推出真实瓶颈；低原成功支持记信息不足，不冒充稳定。具体剂量、阈值、簇区间、工程控制和停止出口见[冻结协议](../../../../experiments/pig_appearance_stability_20261103/PROTOCOL_V1.md)。不因结果改变条件或扩量。

## 实际结果与判断（2026-11-03）

[工程1158与正式1159](../../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)均已完成。真实原图／bright／dark成功数1460／1458／1460，预定最坏损失2／1460＝0.136986%，簇区间[0,0.395849%]；合成A/B均278成功、逐对差异0／278，全部280owner有有效支持。两侧均低于原5%／10%效应线；未调剂量、样本、阈值或checkpoint。

因此关闭“本范围出现大幅粗粒度实例成功数损失”这一具体入口，判断只在冻结范围内为不支持。它不是对全部成像可靠性或受控合成价值的否定。原图在已见TRAIN、IoU50和低输出阈值下触顶，AP、细边界、置信度排序、新条件以及更强变化仍未知。预览阈值0.25的事后说明单独标记，不回填为事前终点或挽救原Gate。

## 2026-11-04综合反馈：固定模型稳定性与训练保持分开

本试验的N是[多阶段训练后的项目保留权重](../../PIG_YOLO_BASELINE_LINEAGE_20261104.md)，校正称呼不改变实际输入、结果或Gate。[综合复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#6-三个首试没有方法成绩的地方保留未测)将“固定N在本组变化下粗粒度成功数较稳定”与“更新N后是否损伤能力”分开；夜间保持项的相对作用不能当作本成像假说的阳性补证。

本具体Idea继续`closed/contradicted`，仅限预定大幅效应；[覆盖迁移Idea](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)及[跨条件缺口](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)仍保留AP、精边、新来源和强变化未知。该范围已回到[泛化纲要](../../../outlines/OUTLINE_PIG_GENERALIZATION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，没有新增评价或重开保持训练。

16份冻结源、54项产物、158个原标签文件均核SHA；444个IoU矩阵由原标签与完整预测独立重算，误差0，另一种最大流算法及两侧区间复算一致。实际合计56GPU秒／224CPU-core秒、476前向、0优化。当前不启动仅依据这组粗粒度损失的保持方法训练；继续应先有更接近真实失效的新证据。
