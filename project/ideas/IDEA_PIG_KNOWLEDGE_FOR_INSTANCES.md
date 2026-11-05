---
schema_version: 1
id: IDEA_PIG_KNOWLEDGE_FOR_INSTANCES
kind: idea
title: 获得并利用猪体知识，改善接触与遮挡中的实例归属
summary: 研究猪体知识的获取、真实迁移和实例使用；已有具体版本分别保留取舍，广义效用仍未知。
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

## 主张与问题范围

探索模型怎样获得、保存并利用关于猪的知识，改善接触与遮挡中的实例归属。知识获取、真实域可读性、实例使用和最终效用需要分别解释。本Idea由[SPM主线](../../paper/manuscripts/spm_yoloseg/README.md)组织已有探索，是[中心命题](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)中的一个方向；关系、观察、监督和场景迁移仍可独立发展。

## 当前判断及其依据

广义判断继续未知。已有证据使具体主张发生分化：

- [固定模型消融](IDEA_PIG_C21_V23_ABLATION_V1.md)支持内容、细尺度、外观和选择保护的实际作用。问题进一步落到内容总量、正确空间／部位对应与选择器响应的分工。
- [原生部位V1](IDEA_PIG_NATIVE_PART_V1.md)、[RLPL](IDEA_PIG_RLPL_V2.md)及[同COCO起点联合学习](IDEA_PIG_COCO_JOINT_V1.md)分别限制各自语义或对应配置。一般辅助作用、语义特异增量和相对保留模型的适配收益需要按匹配比较理解。
- [V23共享保留](IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)获功能简化支持，实际计算收益待测；[C21局部保持](IDEA_PIG_C21_FINE_PRESERVATION_V1.md)结束当前扩训。[固定池质量头](IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)继续保留原关闭范围。
- [分配—部位—Mask](IDEA_PIG_ASSIGNMENT_PART_MASK.md)仍暂停，正常评价未完成，科学判断未知。

## 哪些认识会改变选择

值得继续辨别的是：哪些额外信息超出已有实例监督或非语义控制，能否迁移到真实对象，正常输出是否实际利用它们，以及共同适配为何退化。新的信息或机制证据应对应其中的具体缺口；已有配置的失败仍约束后继选择。

方法与经验由[辅助学习知识](../skills/multi-task-optimization/references/auxiliary-transfer.md)和[对象读出知识](../skills/yoloseg-method-design/SKILL.md)维护。[外部学习记录](../research_notes/literature/pig-auxiliary-transfer_2026-11-05/README.md)保留当时适用性分析，新方法的真实效用尚未检验。

当前行动与暂停事项见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)保留原型、对象、部位及各轮反馈的演变和来源。
