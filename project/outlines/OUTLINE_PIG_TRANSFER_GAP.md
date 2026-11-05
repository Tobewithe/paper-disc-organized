---
schema_version: 1
id: OUTLINE_PIG_TRANSFER_GAP
kind: outline
title: 覆盖不足、场景依赖和适配损伤怎样区分？
parent_id: OUTLINE_PIG_GENERALIZATION
outline_level: 3
summary: 围绕跨条件失败比较缺失数据、可用场景信息和更新造成的能力损伤，保留独立来源与成本的未知。
author: AI依据三层问题结构与既有研究整理
curation_status: source_based_draft
evidence:
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 已完成首个冻结模型条件干预，两侧均低于预定大幅效应线
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: SAP、DAFormer与简单重组的监督、优化和迁移条件
    relation: references
---

## 具体缺口与已有认识

同场景有效而新条件失败，可能因为没见过关键组合，也可能依赖偶然背景，或者适配破坏了原本能力。这些解释对应不同投入：补覆盖、获得场景信息、改善监督或约束更新。

## 已排除范围与未知

**2026-11-05更新：** [共同COCO起点与局部保持训练](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)把部分问题分得更清楚：R／Q／C在5→20轮的同源开发AP均下降，检测AP也走弱；更换起点没有自动消除普通适配退化。C21仅开放5201个细尺度参数时退化较小，但在同一范围增加KL仅+0.024948、成功状态变化0／−1／+1，本版扩训结束。

这些结果来自当前真实任务与反复使用的432dev。局部范围的稳定性、KL的边际作用、训练泛化退化与新场景覆盖是不同对象；后续优先辨别共同训练条件和额外信息的作用，保留[覆盖迁移Idea](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)的新条件问题，反馈[泛化纲要](OUTLINE_PIG_GENERALIZATION.md)与中心纲要。

[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)和[原生部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)本版停止，但未唯一定位迁移原因。单一身体／环境家族的合成资格不等于覆盖，重复开发集也不提供独立泛化证据。可用新来源与其拆分身份仍待确认。

## 首个实际检验留下的认识（2026-11-03）

[成像敏感性V1](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)固定同一模型与几何：真实128 TRAIN／1460 GT在两种预定色调操作下最坏实例成功数损失2／1460＝0.136986%，簇区间[0,0.395849%]；30组合成A/B在原owner有效域均278成功、逐对差异0。基线支持充足，但均低于原5%／10%效应线，具体入口结束，暂不训练保持方法。

这区分了“可以控制成像因素”与“该因素是当前粗粒度实例能力瓶颈”。原图在已见TRAIN的IoU50、低输出阈值计数上触顶，不能据此推断AP、细边界、分数排序或新相机稳定。温和照明和单一家族也不能代表强暗光／噪声或所有场景变化。覆盖、几何上下文及适配损伤仍未被这次检验排除。

后续优先需要实际新条件或细粒度失效证据，判断到底缺观察、实例表示还是可靠排序；不以加大gamma、继续堆数据或附加保持loss作为默认后继。原失败与这次阴性各自保留，不拼为唯一域差因果。

## 2026-11-04综合认识回流

[夜间Idea复盘](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)确认P比同条件普通R高2.515091 AP、比D高4.105891，却仍比多阶段保留N低2.094301、损伤4.4685%。因此保持约束有相对作用，安全适配仍未实现；[覆盖迁移Idea](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)继续未知，不能从该结果声称新场景能力已得到验证。

[综合总结](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)同时限定归因：RLPL有合成BN曝光，概念／夜间BN完全冻结仍退化，BN不足以作为统一解释。固定N的温和成像稳定性、更新N后的能力保持、开发外迁移是不同问题；这些区别反馈到[泛化纲要](OUTLINE_PIG_GENERALIZATION.md)和[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，不以新解释恢复旧配置。

## 关联Idea与可能方向

[覆盖、迁移与保持](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)可以考虑少量真实弱标、稳定场景参考或更新策略；[受控合成](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)可以帮助区分外观与几何依赖。真实场景本身也提供信息，不能把所有背景抹掉就称稳健。

每次结果应回写[泛化问题](OUTLINE_PIG_GENERALIZATION.md)：具体支持了哪个解释，哪个投入因此降低优先级，哪些可能性仍没有被检验。
