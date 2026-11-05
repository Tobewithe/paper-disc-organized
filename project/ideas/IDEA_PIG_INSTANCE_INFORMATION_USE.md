---
schema_version: 1
id: IDEA_PIG_INSTANCE_INFORMATION_USE
kind: idea
title: 可见片段怎样组成一头猪：实例信息的表示与利用
summary: 探索同体片段归组与异体分离；当前缓存支持不足，广义表示效用仍未知，保留新观察与关系监督的可能。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES]
line_ids: [LINE_PIG_FRAGMENT_RELATION_20261103]
exploration_status: paused
assessment: unknown
author: AI依据用户澄清及已核文献作开放构思
curation_status: source_based_draft
evidence:
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 具体数据入口支持不足，未检验广义表示或正常分割效用
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: CClusnet、MP/MGB、TAPPS与多架构猪基准；正面相关依据及已有做法
    relation: references
---

## 想法与价值

将实例理解为属于同一头猪的一组可见证据，探索怎样跨不连通区域保持同体、在相邻身体间保持区分。中心证据、关系归组、对象查询、片段层次或部位线索都可作为切入，具体表示取决于可用信息。

[CClusnet等相关证据](../research_notes/literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md)支持片段问题值得研究，但已有归组方法本身不构成本项目贡献。[选择性关系标注](IDEA_PIG_RELATION_ANNOTATION.md)可进一步追问模型缺哪种关系、额外标注是否提供原Mask之外的信息。

## 当前判断与未决问题

广义Idea暂停、判断未知。[TRAIN适用性首试](IDEA_PIG_FRAGMENT_RELATION_PILOT.md)结束当前160网格缓存入口：可靠原断连支持不足，剔除重叠诱发的断片不能替代物理可见监督。方法效用、归组错误占比和人工成本尚未检验，原始结果见[报告](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。

既有对象／部位接口和[概念V1](IDEA_PIG_NATIVE_PART_V1.md)的结果降低了对具名语义必需性的信心，其他关系组织形式仍开放。若原始观察不足，需要比较[额外观察](IDEA_PIG_OBSERVATION_INFORMATION.md)或[按需细节](IDEA_PIG_ADAPTIVE_DETAIL.md)；若原Mask已有足够关系监督，新增标签可能冗余。

更明确的可见标注、可核序列或真实归属错误可提供继续依据。探索由[片段关系主线](../../paper/manuscripts/pig_fragment_relation/README.md)组织，行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_INSTANCE_INFORMATION_USE.md)保留原局部候选与首试反馈。
