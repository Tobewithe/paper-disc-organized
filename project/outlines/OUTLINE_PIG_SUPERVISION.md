---
schema_version: 1
id: OUTLINE_PIG_SUPERVISION
kind: outline
title: 哪些监督和知识最值得获得？
parent_id: OUTLINE_PIG_INSTANCE_SEGMENTATION
outline_level: 2
summary: 比较已有Mask、少量关系标注、语义、合成与对应提供的信息及成本，不预设某类辅助有效。
author: AI依据用户三层结构要求及既有研究整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 猪框监督、合成动物部位、普通实例训练与受控数据的条件
    relation: references
---

## 关键问题

现有独立Mask已经提供很多实例信息。值得研究的是哪些额外标签、关系或观察真正补足缺口，哪些只是换一种组织已有监督，以及有限人工与计算应该用在哪里。

## 已有认识与排除范围

**2026-11-05新增证据：** [同COCO起点联合训练](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)中，正确部位C比无辅助R平均高1.960429 AP，却比非语义Q低0.513151且seed异号；Q−R为+2.473579。相对普通训练有效的额外监督，并未在这个配置中表现为正确部位的稳定特异收益。从强N续训改为COCO起点仍未解决该增量，当前原生部位方案停止扩训，保留一般辅助效应和长程适配退化。

与此同时，同批V23功能简化保持了内容重建收益，共享保留三seed相对完整重训均提高。现有内容／外观与受保护读出的正面证据、原生五部位辅助的阴性，以及正确对应仍开放的知识问题，分别进入[学习缺口](OUTLINE_PIG_TRAINING.md)、[表示缺口](OUTLINE_PIG_MODEL.md)和[广义猪知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)。

[RLPL V2](../ideas/IDEA_PIG_RLPL_V2.md)相对连续控制有优势，却低于共同合成曝光的无辅助B2；[原生部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)正确部位C未稳定超过等容量非语义Q。两版未满足净效用和保护，继续关闭；其参照N是多阶段项目保留权重，不能被误称为未经目标训练的官方模型。

[本轮综合反馈](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)进一步区分了额外监督的相对作用和模型采用价值：[夜间P](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)相对普通R有+2.515091 AP，但仍不足以替换N。辅助头与原生模块有实际更新，不代表概念已充分迁移并被使用；普通适配、语义必要性和任务利用继续作为[学习缺口](OUTLINE_PIG_TRAINING.md)中的竞争解释，[猪体知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)仍未知。

[TRAIN片段首试](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)仅找到14个可靠原断连支持，停止用当前缓存直接训练显式断片关系。它提醒：原独立Mask不一定编码物理可见断片；当前并没有比较新增关系标签的效果或实测人工成本，因而关系标注Idea仍开放。

[C21／V23本轮功能消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)进一步发现当前重建器对空间内容、细尺度和外观对应有实际依赖，选择策略保护已有正确预测。知识研究由此更聚焦于“内容总量、正确对应及任务选择各起什么作用”；[学习缺口](OUTLINE_PIG_TRAINING.md)保留这一新判别问题，[广义猪知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)吸收局部正面证据，整体判断仍未知。已有部位／RLPL结果继续约束各自监督方案。

## 具体未解缺口

- [哪些少量标注能解除实例归属歧义？](OUTLINE_PIG_REAL_DATA.md)
- [受控合成能提供哪些缺失信息与可区分证据？](OUTLINE_PIG_DATA.md)
- [正确语义、普通关系监督与适配分别提供什么作用？](OUTLINE_PIG_TRAINING.md)

## 从这里形成的Idea

[选择性关系标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)、[受控合成认识工具](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)与[猪体知识](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)是不同切入。模拟标签数不能代替实测人工成本，合成可预测也不能替代真实效用。

具体想法由主线探索后，把新增信息是否有用、哪些解释被削弱及仍未知什么回写到本页和对应缺口，并回到[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

## 2026-11-05反馈循环后的监督判断

A组的正确部位监督没有稳定超过非语义Q，说明相对普通辅助的改善不能直接称为猪体知识的实例价值；C21局部保持也没有形成可用的新增决策作用。当前应保留已有内容／外观／选择保护的正面证据，把语义迁移和局部保持放入“需要中间环节诊断后才重开”的状态。下一循环优先复用A预测定位训练退化，暂不恢复旧分配矩阵或新增部位训练；只有新证据显示额外监督被真实域读取并被实例输出使用，才重新冻结小规模比较。
