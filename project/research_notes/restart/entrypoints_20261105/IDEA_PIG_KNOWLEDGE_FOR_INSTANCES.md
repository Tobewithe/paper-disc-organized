# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_KNOWLEDGE_FOR_INSTANCES
kind: idea
title: 获得并利用猪体知识，改善接触与遮挡中的实例归属
summary: 追踪原型、对象与部位知识怎样被学习并实际用于实例分割，保留竞争解释与思路演变。
outline_ids: [OUTLINE_INSTANCE_SEGMENTATION, OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_ORGANIZATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_HYPOTHESES, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: unknown
assessment: unknown
author: AI 整理，依据项目 Idea 演变与研究资料
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/idea_evolution_e93bf5358866376b.txt
    locator: 原文 project/research_notes/IDEA_EVOLUTION.md
    revision: e93bf5358866376b3bf31ffa649bf6fe01ba74e51a79d85866c8a9cdfaecc893
    relation: references
  - path: project/research_notes/sources/pig_instance_analysis_e4e6e8d0e8969e75.txt
    locator: 原文 literature_search/pig-instance-open-analysis_2026-11-02/summary.md
    revision: e4e6e8d0e8969e7504bd9ad4bfc671b14f66255ec0987adaebc8aadd9c51de88
    relation: references
---

## 想法与知识问题

怎样让模型获得并利用关于猪的知识，从而区分接触、遮挡中的不同实例？知识保存在哪里、怎样学到、怎样用于任务，是需要分别回答的问题。

可复用的方法认识见[多任务Skill的辅助学习正文](../../../skills/multi-task-optimization/references/auxiliary-transfer.md)。本Idea继续记录猪体知识这一具体主张的依据、检验和判断，通用知识在Skill中维护，其保存不以该Idea当前是否成功为前提。

**2026-11-05外部学习接续：** [方法学习与当前适用性检验](../../literature/pig-auxiliary-transfer_2026-11-05/README.md)把这三层问题接到具体技术。Auto-Lambda／ForkMerge以主任务效用选择辅助更新；SAP的真实域伪标签按精度而非仅按稀有度加权；TAPPS将对象身份写入部位目标。它们分别处理辅助价值、迁移与使用，不能互相替代。当前C未稳定胜Q、无辅助也退化，故保留非语义控制及共同训练解释；已有实例条件头的阴性不因读到TAPPS而重开。广义判断仍unknown，方法已进入两个现有Skill，真实效用检验尚未运行。

## 已有探索与演变

[主要 Idea 演变轨迹](../../sources/idea_evolution_e93bf5358866376b.txt)记录了视觉原型字典、身体结构、具名部位记忆、内容参与重建、共同坐标、训练期对应和概念监督等变化。这些分支与回转保留原有依据，不能将它们压成一条所有前案都被否定的直线。

[SPM 研究主线](../../../../paper/manuscripts/spm_yoloseg/README.md)继续作为已有探索入口。该条目只连接其知识问题，不把旧目录名称固定为未来方法选择。

## 与新结构和最新实验的关系

本Idea保留“猪知识怎样帮助实例任务”的历史探索问题，不承担全部猪实例分割分析的标题或默认结论。[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)容纳实质不同的表示、观察、监督、合成、迁移与可靠性Idea。后续探索可以借鉴本线，也可以独立发展，不因本条关联SPM就限定为概念训练。

最新[分配—部位—Mask有限判别](../../../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)单独记录为已暂停、研究判断未知：2026-11-03补核CP/QP后，六臂各2000末点元数据均已核，dev评价仍未完成。这样既保留原概念探索的真实进度，也不由某一暂停或阴性版本裁定整体猪知识Idea。

## 用户澄清后的定位（2026-11-02）

上一轮把本线的“结构监督如何产生独立真实收益”提升为总问题，又据此组织论文，收窄了用户需求。旧论文草案仅保留为一个局部思路的历史材料；它的三候选排序和A/B二选一安排不约束中心纲要的全部探索。

V21/V23的正常收益与内容依赖不足、RLPL／概念V1的本版净效用失败、V32竞争相对变化与错位近乎无损，仍应反馈到本线的信心和选择；它们不能排除关系型监督、真实历史观察、场景迁移等不同问题。新的[开放探索图](../../../outlines/OUTLINE_PIG_HYPOTHESES.md)连接这些可能性，不要求先把本线做成。

深化本Idea仍有意义，例如追问什么知识真正超出已有实例监督、怎样迁移到任务；其价值与其他方向平行比较，而非默认最优先。原概念Goal及正式Gate只在所属执行中保留，不能成为未挂线想法的内容门槛。

## 具体Idea的取舍与反馈

以下具体版本已有停止依据，并已回写[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)：[固定池质量头V25／V26](../../../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)、[RLPL V2](../../../ideas/IDEA_PIG_RLPL_V2.md)、[原生部位辅助V1及受限适配后继](../../../ideas/IDEA_PIG_NATIVE_PART_V1.md)。它们各自保存原主张、结果、排除范围和仍开放的问题，不将局部失败直接赋给整个猪体知识Idea。

[当前分配矩阵](../../../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)仍暂停、判断未知。后续新认识继续在具体Idea及相关纲要中保留，主线组织其探索，证据与取舍再反馈到知识；本大Idea不因旧版本关闭自动改判。

## 2026-11-04跨实验认识回流

[综合复盘](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)纠正了比较身份：项目保留N经过多阶段训练，方法增量应优先看各自普通控制。对本问题最直接的现有约束仍是[部位V1](../../../ideas/IDEA_PIG_NATIVE_PART_V1.md)的C−Q小且seed异号，以及[RLPL](../../../ideas/IDEA_PIG_RLPL_V2.md)的D2低于共同曝光无辅助B2，不能用另一对更差控制的正差声称猪体知识已被有效利用。

审计排除了“只训练新头”等简单解释，但尚未区分知识迁移不足、任务利用不足与适配条件，不能因此复活已关闭版本。对猪知识的广义判断继续未知；这些限制已反馈到[监督问题](../../../outlines/OUTLINE_PIG_SUPERVISION.md)、[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)和[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，其他观察、表示和监督方向仍平行开放。

## 2026-11-04固定模型消融带来的新认识

[本轮七项消融](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)及独立验收给出了内容使用的正面证据：C21重建器内容输入置零下降2.306973 AP，细尺度残差置零下降2.565348，V23外观码置零下降1.315483。当前读出对空间内容、细尺度修正和外观对应存在实际依赖；强制完整候选造成大量损伤及空Mask，更支持内容重建与选择保护共同发挥作用。

这使本Idea的具体研究重点从笼统追问“内容是否被使用”，推进到“内容总量、正确空间／部位对应和选择器响应各起什么作用”。下一判别优先保持内容总量，比较正常动作重选时的正确对应与等量错配。广义猪知识的判断仍为未知，已有正收益模型保留；形变和分部位保留在本次干预中作用较小，进入简化候选。旧概念、RLPL与固定池扩展各自保留原决定。

上述认识沿[具体消融Idea](../../../ideas/IDEA_PIG_C21_V23_ABLATION_V1.md)进入[实例表示](../../../outlines/OUTLINE_PIG_MODEL.md)、[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)、[监督问题](../../../outlines/OUTLINE_PIG_SUPERVISION.md)与[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)；主线优先级同步更新。

原分配矩阵继续暂停。

## 2026-11-05反馈循环后的广义判断

夜间三组具体版本已分别停止、保留或结束，但它们只改变了对应配置的优先级：一般辅助可能有效，正确部位的特异价值尚未证实；共享读出约束有功能精简依据，实际计算收益未测；局部保持的新增决策作用很小。广义猪体知识仍保持`unknown`。后续必须沿“学会—真实域可读—实例使用—最终效用”逐段提供证据，不能用任一辅助loss、梯度可达或局部AP差替代整条链。下一循环优先复用既有预测定位训练退化，再剖析共享／形变分支的真实成本；在这些信息出现前，不重开A/C语义训练或旧分配矩阵。

## 2026-11-05三个新版本验收后的认识

[夜间三组Idea完整报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)已完成三seed训练、完整432dev、独立统计和最终包验收。[同COCO起点联合学习](../../../ideas/IDEA_PIG_COCO_JOINT_V1.md)的正确部位C−Q平均−0.513151且异号，C−R虽+1.960429，非语义Q−R更高；更换学习起点没有带来稳定语义增量，当前版本关闭，保留一般辅助作用和长程适配退化。

[V23功能简化重训](../../../ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)将原低敏感性推进为重训后可维持的证据：形变置零+0.010234、平均共享保留+0.148434，相对同预算完整控制；共享三次正向、两种简化均过护栏。优先保留共享读出用于实际结构压缩，原V23继续作成熟参照。[C21细尺度保持](../../../ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md)则只有+0.024948、成功状态变化0／−1／+1，结束本版扩训，保留局部更新较稳定的认识。

已有内容／外观／选择保护的正面结果保留，原生辅助的语义特异价值与内容正确对应仍分别是开放问题。广义猪知识继续未知；本次关闭的具体版本、功能简化的支持及仍需辨别的获取／迁移／利用环节，进入[学习](../../../outlines/OUTLINE_PIG_TRAINING.md)、[表示](../../../outlines/OUTLINE_PIG_MODEL.md)、[监督问题](../../../outlines/OUTLINE_PIG_SUPERVISION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。
