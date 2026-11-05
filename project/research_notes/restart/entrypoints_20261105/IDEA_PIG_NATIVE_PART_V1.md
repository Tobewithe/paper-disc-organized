# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_NATIVE_PART_V1.md](../../../ideas/IDEA_PIG_NATIVE_PART_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_NATIVE_PART_V1
kind: idea
title: 原生可见部位辅助V1及受限适配后继（已停止）
summary: 四臂三seed未得到稳定概念独立增量，整体收益和保护失败；预注册重放也未支持受限适配V2，不继续本配置。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
author: AI于2026-11-03据原正式报告及重放回填具体Idea身份；非新增实验或事前注册
curation_status: source_based_draft
scope:
  claim: 原V1全非BN适配下正确合成部位能相对Q提供稳定且满足正常效用／保护的额外收益。
  boundary: 原72训练／18QA辅助语料、2570参数接口、2000更新、三配对seed及预注册读出重放。
  retrospective_record: true
evidence:
  - path: experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md
    revision: 71fdc28133cda7a23fe99155240aea6d7343e9d5d3d059d4e06e5e0a9c6d4039
    locator: 四臂三seed正常评价、损伤及关闭V1和受限适配后继的依据
    relation: references
  - path: experiments/spm_concept_native_training_20261102/READOUT_REPLAY_V1.md
    revision: dba9401c912dd2ba6db943916902e69c8d339ca85012bd7a8f8f53391940be29
    locator: 预注册重放范围与受限适配V2的必要条件，未执行V2训练
    relation: references
---

## 当时想验证什么

在原生非BN参数适配中，真实原GT保持正常监督，合成提供正确五部位C，对照真实续训R、几何辅助G和等容量非解剖空间分区Q，检验正确概念能否提供额外真实收益并保住原成功实例。

这是据原报告回填的限定想法记录，固定版本、协议和Run仍沿原文件，当前记录不冒充事前注册。

## 实际结果与停止理由

[V1正式报告](../../../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md)：R／G／Q／C各三配对seed、每臂2000更新及完整432dev评价完成。C平均Mask AP56.494576，较N57.343500低0.848924点；C−Q仅0.088211点且seed异号。C原成功Mask损伤64／67／66÷2126，约3.01%—3.15%，均超过原1%线。

关闭V1采用与追加训练，正常效用、保护及稳定概念独立增量的联合主张未成立。有限重放固定N检测并用无GT框关联后，C平均57.384169，仅N＋0.040669、Q＋0.012135，损伤仍超限；预注册的受限适配V2必要条件失败，故不启动该后继训练。

这里包含两种事实：V1有真实训练阴性；V2因前置必要条件失败被排除，**没有V2训练结果**。不能把两者写成两个均已训练失败的方法。

## 排除范围与保留问题

停止原V1及其受限适配后继，不据此判定所有猪体知识无效。合成QA可读也不证明真实概念充分迁移；该实验未唯一识别语义冗余、迁移不足与正常适配损伤。

[实例片段关系](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[选择性关系监督](../../../ideas/IDEA_PIG_RELATION_ANNOTATION.md)、[额外观察](../../../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)分别提出不同的信息组织或来源。它们尚未验证，不是为旧V1补loss的授权。[当前分配—部位—Mask矩阵](../../../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)改变了接口和分配，属于另一具体版本，仍暂停、未裁决，不能继承本条阴性。

重新考虑部位方向需要有根据的新前提或直接作用证据；不能只用“合成loss更低”“多训练一些”把已关闭配置当作未尝试。原概念Goal保留，后续命题和执行按各自实际范围处理。

## 2026-11-04综合反馈：语义增量与保留模型效用分开

[本轮复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#4-概念监督v1正确部位没有稳定超过非语义控制)补清：N是多阶段训练与分支组合后的项目保留YOLO权重，R是同N起点的普通真实续训。C−N低于零说明本配置不能替换N；判断部位语义增量仍看C−Q仅+0.088211且seed异号，不能混称“概念训练不如COCO初始模型”。

原生非BN模块和辅助头实际更新、BN固定的审查结果，削弱了“只训练新头”及“BN漂移统一造成失败”的解释；它没有替本版稳定语义增量的阴性翻案。同COCO起点的目标任务联合训练尚无对应直接比较，属于未测条件，不自动成为重开V1或V2的依据。

本具体Idea继续`closed/contradicted`，本轮没有新实验。认识回到[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)、[监督信息价值](../../../outlines/OUTLINE_PIG_SUPERVISION.md)及下方中心入口；广义[猪体知识Idea](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)仍未知。

## 回到纲要

[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)保存这项排除及其边界，[开放探索](../../../outlines/OUTLINE_PIG_HYPOTHESES.md)连接仍未解决的问题。暂停、test426隔离与原Gate保持。
