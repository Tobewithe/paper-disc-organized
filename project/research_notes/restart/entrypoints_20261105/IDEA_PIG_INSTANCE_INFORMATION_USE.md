# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_INSTANCE_INFORMATION_USE
kind: idea
title: 可见片段怎样组成一头猪：实例信息的表示与利用
summary: 从同体片段归组和异体分离研究实例能力，允许中心、关系、对象表示与可选部位线索，不预设当前概念训练方案。
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

## 核心想法与大命题

[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)关注真实猪场的逐头实例分割。本方向把实例理解为一组属于同一头猪的可见证据，探索怎样跨不连通区域保持同体、在相邻身体间保持区分。它研究表示与信息组织，不先要求重建完整身体或学习五个解剖名称。

可能的切入包括片段—对象的层次表示、关系／亲和归组、中心证据、对象查询或局部结构。是否需要显式片段、何种关系可辨，都是开放选择，不把一个具体网络预先命名为答案。

## 依据及值得探索的价值

[CClusnet证据卡](../../literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md)有猪仔遮挡断片归组的直接相关证据；多架构猪基准也提示高总体分数与细部归属不是同一能力。该问题可能连接同体拆分、邻猪合并及肢体错归，并可与[选择性关系标注](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)结合：先确定缺哪种关系，再思考最少需要怎样教给模型。

中心投票、Mask引导、对象—部位绑定都有前作。潜在价值来自新的问题理解、合适的信息或监督组织及其真实作用，不在于重做这些模块或给“关系”换名。

## 与既有主线的联系及区别

[V32](../../../../experiments/spm_registered_object_head_20261027/PROTOCOL_V1.md)的竞争相对小收益、部位错位近乎无损，与[概念V1](../../../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md)的C−Q不稳定，要求降低对具名语义必需性的信心；它们没有排除其他片段归组形式。

上一轮提出的“合成部位×真实相对归属”保留为本方向一个局部、未验证的方案。其C/Q与独立点监督控制只服务于那个语义交互主张，不能定义本Idea外延，也不要求新的关系标注、片段表示或其他输入路径都经过它。

## 未知与当前判断

本项目尚未证明归组是主要错误来源。若可靠片段本就不存在，或原始观察不足，继续重组同一信息可能无益，应考虑[额外观察](../../../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)或[按需细节](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)。若完整Mask已有足够监督，新增关系标签也可能冗余。

## 当前探索与下一层具体想法（2026-11-03）

用户要求依据Idea开设主线做实验后，本Idea进入[片段关系主线](../../../../paper/manuscripts/pig_fragment_relation/README.md)，具体首试保存为[TRAIN适用性Idea](../../../ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)。该实验已完成：14642个原独立Mask仅14个提供可靠断连支持，低于预定128实例／32图；150个额外断连由剔除重叠诱发，不是可直接替换的可见真值。见[完整报告](../../../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。

当前缓存的显式断片训练入口因此结束；本广义Idea暂记“已暂停、判断未知”，没有被方法效用实验反证。原标注连通不能证明图像证据也连通，数据实验没有测出归组错误占比或标注成本。更明确的可见标注、原图语义或真实错误证据可以改变选择，目前没有具备条件的训练后继。
