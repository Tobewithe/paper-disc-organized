---
schema_version: 1
id: IDEA_PIG_SYNTHETIC_COUNTERFACTUALS
kind: idea
title: 用受控合成发现模型依赖与实例混淆的来源
summary: 用物理一致的成对变化辨别模型依赖；首个成像解释受到约束，其他干预及真实外推仍待依据。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_GENERALIZATION, OUTLINE_PIG_TASK, OUTLINE_PIG_DATA, OUTLINE_PIG_MODEL, OUTLINE_PIG_HYPOTHESES]
line_ids: [LINE_PIG_CONDITION_RELIABILITY_20261103]
exploration_status: paused
assessment: unknown
author: AI依据现有可控合成能力及用户开放探索要求推导
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 有限release的真实几何、标签资格及资产家族边界；不证明新成对场景已实现
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 合成猪检测、Copy-Paste与SAP显示内容、标签和迁移条件需分开考虑
    relation: references
---

## 想法与价值

利用少量物理一致的场景变化辨别实例混淆来源：几何固定时改变照明或背景，外观保持时改变接触、遮挡或视角，并同步生成图像与相应标签。结果可导向表示、真实标注或额外观察，最终方法可以采用不同训练途径。

与[片段表示](IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[覆盖迁移](IDEA_PIG_COVERAGE_TRANSFER.md)及[可见证据](IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)的联系在于生成有区分力的竞争解释。受控渲染和反事实分析已有前作，本项目的贡献与真实适用性需要对应证据。

## 当前判断与范围

后续执行暂缓，整体判断未知。[成像敏感性V1](IDEA_PIG_APPEARANCE_STABILITY_V1.md)使用既有照明配对和真实色调变化，未支持本范围的大幅粗粒度能力损失。它约束了一个具体解释，原结果见[报告](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)。

单一资产家族、温和变化、已见TRAIN与IoU50口径限制了外推。其他成对干预尚未由原release证明已实现，新的真实失效或可辨别解释可以支持后继选择。生成、几何标签与分布知识由[合成场景Skill](../skills/synthetic-scene-methods/SKILL.md)维护。

探索由[条件可靠性主线](../../paper/manuscripts/pig_condition_reliability/README.md)组织，当前行动见[当前安排](../ACTIVE_PLAN.md)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)保留原例子、限制及首试反馈。
