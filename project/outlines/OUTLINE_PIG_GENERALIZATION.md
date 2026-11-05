---
schema_version: 1
id: OUTLINE_PIG_GENERALIZATION
kind: outline
title: 怎样在新条件下保持能力？
parent_id: OUTLINE_PIG_INSTANCE_SEGMENTATION
outline_level: 2
summary: 连接条件覆盖、场景信息、知识迁移与适配保持，区分同源开发收益与新场景能力。
author: AI依据用户三层结构要求及既有研究整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: Copy-Paste、SAP、DAFormer与猪场数据划分的迁移条件
    relation: references
---

## 关键问题

场景、相机、光照、体型或生长阶段变化后，什么能力应该保持，什么信息可以少量更新？数据覆盖、有效场景上下文与偶然捷径需要区分，训练更新本身也可能侵蚀正常能力。

## 已有认识与排除范围

当前开发集被反复使用，来源有限；增加seed不能把它变成独立场景确认。有限合成release具有标签资格，但不代表真实猪场覆盖。[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)和[部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)的适配损伤提示必须保留正常能力，但没有证明单一域差或BN解释。

[成像敏感性首试](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)已给出一个具体限制：128个真实TRAIN图在固定色调操作下的实例成功数最坏损失仅0.136986%，30组固定几何照明对照差异为0，未达预定效应线。这降低对“本范围外观引起大幅粗粒度实例能力损失”的信心，当前不据此训练保持方法。真实已见来源、IoU50和低阈值触顶，尚不能排除细边界、置信度或独立新条件风险；数字色调与物理照明也不混作同一干预。

## 2026-11-04综合反馈

[保持训练的相对作用](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)没有改变上述成像首试的效应判断：前者更新模型，后者固定模型，均未提供新场景确认。P相对R改善但仍损伤保留N，加上BN固定的实验也退化，使“保持能力”值得继续作为问题，而非已经解决或统一归因于BN；[覆盖迁移Idea](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)仍暂缓／未知。

[综合总结](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)的这项区别已回到[覆盖／适配缺口](OUTLINE_PIG_TRANSFER_GAP.md)及[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。同源开发成绩不能代替新域证据，也不据此自动增加gamma、辅助loss或训练规模。

## 具体未解缺口

**2026-11-05反馈：** [夜间三seed结果](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)来自已有432dev：COCO起点也出现随训练推进的开发退化，C21局部更新较稳定而KL增量很小。当前降低继续延长这些配置、用保持项直接获取新增AP的优先级；普通适配与新条件覆盖分别保留为[具体缺口](OUTLINE_PIG_TRANSFER_GAP.md)。真实新来源和跨场景失效分母仍是独立问题。

[覆盖不足、场景依赖和适配损伤怎样区分？](OUTLINE_PIG_TRANSFER_GAP.md)

## 从这里形成的Idea

[覆盖、迁移与保持](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)可以比较真实弱标、场景参考、教师、合成和适配；[受控合成](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)可以帮助提出区别解释。未来研究不必依赖部位语义成功。

新条件和新来源是否可得仍未知。正式验证前明确源、拆分和比较权限；结果按对应范围反馈，不将同场景涨分扩大为跨农场能力。
