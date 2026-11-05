---
schema_version: 1
id: OUTLINE_PIG_TRAINING
kind: outline
title: 正确语义、普通关系监督与适配分别提供什么作用？
parent_id: OUTLINE_PIG_SUPERVISION
outline_level: 3
summary: 比较全／弱监督、辅助任务、可靠视频对应与迁移保持，让不同信息转为任务能力，不以现有概念方案定义学习问题。
author: AI保留学习知识并按三层问题及原实验范围整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 框监督、MP/MGB、SAP和DAFormer
    relation: references
---

## 具体缺口与当前认识

相关成熟思想与方法比较集中在[多任务Skill的辅助学习正文](../skills/multi-task-optimization/references/auxiliary-transfer.md)，其他已有知识由[知识入口](../research_notes/README.md)连接。本纲要维护这些知识对当前研究问题的意义，原阅读范围和实验事实分别保留在来源记录与报告中。

**2026-11-05外部学习补充：** [目标效用与辅助迁移方法](../research_notes/literature/pig-auxiliary-transfer_2026-11-05/README.md)将当前问题接到TAG、Auto-Lambda、AuxiLearn与ForkMerge。学习重点从梯度大小／冲突转到辅助更新对目标数据的实际作用：冲突可能提供有益正则，元目标与有限训练后的效用分别提供不同层次证据。若进入动态辅助新版本，优先低维强度学习，保持C/Q及匹配R；共同训练修正同步应用，避免将对弱基线的修复当成语义增量。SAP附录进一步提示伪标签增权须依目标精度选择，TAPPS则提供对象绑定的监督形式。两个现有方法Skill已实质补强；真实模型检验因SSH阻塞尚未执行，旧科学判断不变。

**2026-11-05更新：** [同COCO起点三seed对照](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)已补齐下方11月4日所记的比较缺项。正确部位C−Q平均−0.513151 AP且异号，C−R虽为+1.960429，Q−R却更高（+2.473579）；三个预定学习点的C−Q均差都为负。当前更支持一般辅助作用，降低“只需换初始化便会得到语义增量”的优先级。R／Q／C随5→20轮推进同时退化，学习侧首先要解释普通任务适配的泛化／排序变化及什么信息真正超出非语义控制。

同批C21仅更新5201个细尺度参数时较原模型平均下降约0.10 AP，加保持项仅提高0.024948、GT75状态变化0／−1／+1；结束本版扩训。局部稳定性和保持的微小边际作用分别保留，旧大范围保持的相对结果仍按原起点和目标解释。该[联合训练Idea](../ideas/IDEA_PIG_COCO_JOINT_V1.md)及[细尺度版本](../ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md)的取舍进入[监督问题](OUTLINE_PIG_SUPERVISION.md)、[适配缺口](OUTLINE_PIG_TRANSFER_GAP.md)与中心纲要。

概念V1和RLPL不能从辅助可学或局部相对优势推出正常净收益。待区分的是新增信息、普通实例关系组织与适配损伤；当前分配矩阵尚未评价，不能填成支持或反证。

[猪体知识](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)、[实例信息利用](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[当前分配判别](../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)分别保存不同范围。具体失败和新结果反馈到[监督问题](OUTLINE_PIG_SUPERVISION.md)，不把同一loss的修补当新命题。

## 1. 不同监督强度下怎样学习？

比较全监督、弱监督、半监督和自监督，分析所需标签、预训练依赖、伪标签误差和样本效率。

## 2. 真实数据与合成数据怎样配合？

比较合成预训练后微调、混合任务训练、合成辅助监督和分阶段适配。研究使用顺序、比例和训练目标如何影响迁移。

## 3. 哪些额外任务值得学习？

比较部位、姿态、深度、边界、关系和对应等辅助目标。区分“辅助任务能够学会”与“最终实例任务得到改善”。

## 4. 训练组织是否比增加信息更重要？

分析样本分配、实例匹配、困难样本、课程、损失平衡与优化稳定性。检验收益是否来自更有效地利用已有监督。

## 5. 怎样处理跨域与适配损伤？

分析教师学习、一致性、域适配及预训练知识保持。区分合成内容不足、域差过大与适配破坏原有能力。

**这里要回答：改善来自新增知识，还是来自数据曝光、训练预算与优化方式的改变？**

## 既有实验带来的认识与限制

[概念V1](../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md)中真实单独续训也退化，BN完全冻结；[RLPL](../../experiments/spm_rlpl_pipeline_20261101/FINAL_REPORT.md)中原型相对连续对照变好却仍低于无辅助与原生。这些事实要求把普通适配、概念相对作用和整体风险分别解释，不能只归因于合成或BN。

[当前六臂矩阵](../../experiments/spm_concept_assignment_bridge_20261102/PROTOCOL_V1.md)区分正确/非解剖监督与三种分配，仍暂停、未评价。C0与旧C的差异还包含容量和loss形式，不能仅由跨版本差值证明实例接口是根因。

学习问题可以从不同入口展开：[少量关系与局部修正](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)提供何种必要监督，[历史观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)能否产生可靠训练对应，[片段表示](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)怎样用好已有Mask，以及[场景适配](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)怎样保留原有能力。它们改变的信息或对象不同，不要求共同使用合成部位辅助。

上一轮C/Q与真实相对归属的交互，仅是语义利用的局部假说；相应控制只服务那个主张，不能成为所有学习Idea的默认矩阵。原真实独立Mask、合成可见标签与unknown仍分别理解，不静默更换旧任务目标。

开放构思先比较哪些信息值得学、为什么可能帮助实例，进入具体方法时再按其主张细化对照。新方向不必先完成原矩阵或真实部位评价才允许讨论；原实验暂停和未决仍按原范围保留。

## 2026-11-04基线身份与联合训练审查

[实际代码／权重审查](../research_notes/PIG_TRAINING_INTEGRATION_AUDIT_20261104.md)区分了被混称“原生YOLO”的对象：当前N是项目多阶段训练和几何／分类权重组合后保留的YOLO26m-seg；R是同N起点的无方法自写循环续训，早期从COCO预训练进行原生结构端到端任务训练的模型另有身份。方法低于N不自动等于低于同起点R，更不能自动等于不如标准YOLO任务训练。

CPU逐tensor核验确认概念V1与夜间版本的主干／颈部／检测／Mask分支均实际更新，辅助头也共同更新；旧RLPL才是部分Mask适配。没有发现所有方法因只训练新头、必要YOLO26 loss漏算而失败的证据。流程／阶段效应和方法增量仍须区分，优先同通用预训练起点的YOLO-only／方法联合训练对照，重大条件匹配即可；该建议未执行，不恢复旧矩阵或替方法翻案。

## 2026-11-04近期实验综合判断

[综合总结](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)的认识已分别进入[部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)、[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)及[夜间训练](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)：P−R=+2.515091说明并非所有方法都低于普通续训；C−Q小且seed异号、D2−B2为负则继续限制各自方法主张。N是多阶段保留权重，低于N用于判断不能替换，不能独立归因于新增方法。

原生与辅助参数已共同更新，排除了“只训练新头”的简单说法；概念／夜间BN固定仍退化，也使BN不足以统一解释。普通适配条件、语义／对应迁移及任务利用仍未区分，同COCO起点的直接比较尚缺；这些是未解问题，不是给失败配置翻案或把所有开放Idea都送入同一训练矩阵。上述认识继续回到[监督问题](OUTLINE_PIG_SUPERVISION.md)、[适配缺口](OUTLINE_PIG_TRANSFER_GAP.md)及[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

## 2026-11-04从已有收益中辨别内容与保护

[C21／V23消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)采用各自固定权重，完整432dev上移除C21内容输入／细尺度分别损失2.306973／2.565348 AP，V23外观码损失1.315483；强制完整候选造成更大损伤。当前学习所得的内容重建、外观对应与选择保护都进入实际输出，后续应优先理解它们怎样配合，而非单纯增加记忆容量或延长旧配置训练。

旧等量内容错配在固定动作／修正量下较弱，新置零沿正常动作重选明显退化，因此新的可区分问题是内容总量、正确对应与选择器响应各自的作用。该问题反馈到[猪体知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)及[监督问题](OUTLINE_PIG_SUPERVISION.md)。原概念／RLPL的监督结论继续保留，分配矩阵评价仍暂停。

## 相关文献线索

[猪框监督研究](https://doi.org/10.1038/s41598-025-05113-x)、[MP-Former](https://arxiv.org/abs/2303.07336)、[MGB-Former](https://doi.org/10.3390/agriengineering8100447)、[SAP](https://arxiv.org/abs/2311.18661)与[DAFormer](https://arxiv.org/abs/2111.14887)对应不同的监督和适配选择。它们的任务、数据和优化条件需要分别核对。

[上层：监督信息价值](OUTLINE_PIG_SUPERVISION.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [相关缺口：可信评价](OUTLINE_PIG_EVALUATION.md)

## 2026-11-05反馈循环的下一判别

A组从5轮到20轮三组均出现Mask／Box下降，三臂共9个配对配置的Mask AP和输出数各自都下降，三seed合计输出量约减少一半；中尺度AP损失明显，大尺度则持平或提高。下一步先比较候选几何覆盖、分数／排序和匹配后的Mask质量，区分共同适配过程与新增辅助目标的作用。若没有跨臂稳定的共同变化，保留未知并结束有限诊断。新版本可由明确的训练／信息瓶颈或额外信息假说进入最小比较，采用则另需配对净效用与损伤支持。范围和退出条件集中见[报告](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。
