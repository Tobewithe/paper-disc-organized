---
schema_version: 1
id: IDEA_PIG_FRAGMENT_RELATION_PILOT
kind: idea
title: 当前真实TRAIN能否支撑可见片段关系研究？
summary: 可靠原断连未满足预定支持量，关闭当前缓存的显式片段训练入口；关系方法与标注成本未测。
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

## 本次检验

为[片段表示](IDEA_PIG_INSTANCE_INFORMATION_USE.md)与[关系标注](IDEA_PIG_RELATION_ANNOTATION.md)核实数据条件：当前真实TRAIN的原独立Mask是否有足够可追溯的非平凡同体断片。原Mask连通性、剔除重叠造成的断连和物理可见性分别解释；算法、支持量与停止出口见[协议](../../experiments/pig_fragment_relation_20261103/PROTOCOL_S0.md)。

## 当前判断与范围

首试结束，判断为支持不足。可靠原断连未达到预定支持量，关闭直接用当前160缓存训练显式断片关系的入口。剔除重叠后新增的断片保留为另一个统计对象，完整分母与核验见[报告](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。

本试验没有关系模型效用或人工标注成本结果。原标签断片少也未测出实际图像中的遮挡和归属错误比例，[广义关系Idea](IDEA_PIG_INSTANCE_INFORMATION_USE.md)继续未知。

高分辨率原标注、明确可见监督、可核序列或真实错误证据可能改变输入条件；新研究需据此说明哪一前提得到补充。实际执行与原冻结版本由[Study](../../experiments/pig_fragment_relation_20261103/study.json)连接，当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)保留数值、原支持线与复盘。
