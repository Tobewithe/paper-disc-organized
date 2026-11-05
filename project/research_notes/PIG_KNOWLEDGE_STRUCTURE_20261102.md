---
schema_version: 1
id: NOTE_PIG_KNOWLEDGE_STRUCTURE_20261102
kind: decision
title: 猪实例分割知识结构整理与已有证据反馈
outline_ids:
  - OUTLINE_INSTANCE_SEGMENTATION
  - OUTLINE_PIG_INSTANCE_SEGMENTATION
  - OUTLINE_PIG_TASK
  - OUTLINE_PIG_LITERATURE
  - OUTLINE_PIG_DATA
  - OUTLINE_PIG_MODEL
  - OUTLINE_PIG_TRAINING
  - OUTLINE_PIG_HYPOTHESES
  - OUTLINE_PIG_EVALUATION
  - OUTLINE_PIG_EVIDENCE
idea_ids:
  - IDEA_PIG_KNOWLEDGE_FOR_INSTANCES
  - IDEA_PIG_INSTANCE_INFORMATION_USE
  - IDEA_PIG_COVERAGE_TRANSFER
  - IDEA_PIG_OBSERVATION_INFORMATION
  - IDEA_PIG_ASSIGNMENT_PART_MASK
coverage: declared
verdict: knowledge_structure_updated_not_a_new_scientific_result
author: AI整理，依据用户查看最新工作台并整理新结构的指令
scope:
  action: knowledge_reorganization
  model_execution: false
  remote_access: false
  study_run_or_gate_changes: false
  inference: no_new_scientific_adjudication
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 已实际阅读的16篇指定方法/实验段
    relation: references
  - path: project/research_notes/sources/concept_native_v1_report_71fdc28133cda7a2.txt
    revision: 71fdc28133cda7a23fe99155240aea6d7343e9d5d3d059d4e06e5e0a9c6d4039
    locator: 已有V1效用/机制阴性及范围
    relation: references
  - path: project/research_notes/sources/assignment_bridge_pause_7e86dc12ee5f9b0e.txt
    revision: 7e86dc12ee5f9b0e34995982c582e925bed004c8880982c4824879780ab065c7
    locator: 用户暂停、四臂末点已核、两臂终态待核
    relation: references
limitations:
  - 本次只组织已有知识，没有新增文献阅读、模型运行或实验裁决。
  - 三个开放候选尚未执行，不能把具体历史Study算作它们的实验结果。
  - 暂停实验尚未评价，未知不记作科学阴性。
---

## 整理依据

最新工作台采用知识纲要、Idea、研究主线、实验与Run的文件声明关系。用户要求查看最新工作台并整理成新结构；本次把猪实例分割开放分析接入这一组织方式，原有实例分割、视频、世界模型和六条研究线身份保留。

## 结构变化

在实例分割纲要下增加[猪实例分割总纲](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，下设八个知识入口。数据、模型和训练各有分析入口，但不提前规定选择；文献证据、竞争解释及评价与反馈相互连接。

三个开放候选分别讨论信息利用、条件覆盖/迁移、额外观察，标记为“候选、判断未知”，尚不绑定研究主线。[猪知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)保留历史演变；具体[分配—部位—MaskIdea](../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)单独关联SPM既有主线，标记“已暂停、判断未知”。

## 已有认识怎样反馈

- 文献支持多个竞争方向，尚不能选定某种骨干、生成器或额外概念为必要方法。
- 旧V1正确部位未稳定超出匹配控制且损伤超线，按原判保留；不能据此否定所有部位知识。
- 最新矩阵四臂完成训练、两臂终态待核，没有正常开发评价，不把计算中断解释为科学失败。
- 数据发布资格、概念可学、任务使用和真实效用继续分账。

原材料以六份新增精确快照保存并在`project/knowledge_sources/SOURCES.json`登记。结构整理不重写协议、Run、预算、test隔离或暂停安排。后续知识更新可通过本记录回到原证据，执行仍查看[总控](../ACTIVE_PLAN.md)。
