---
schema_version: 1
id: IDEA_PIG_VISIBILITY_AND_AMBIGUITY
kind: idea
title: 区分看见、推断与未知的可信实例表达
summary: 从任务语义和不可辨歧义研究可见层、推断层与可信归属，保留原GT含义，不将有限标签线索推广为全数据定性。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_TRUST, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_TASK, OUTLINE_PIG_REAL_DATA, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_HYPOTHESES]
line_ids: []
exploration_status: candidate
assessment: unknown
author: AI依据监督语义差异及用户开放探索要求推导
curation_status: source_based_draft
evidence:
  - path: experiments/spm_rlpl_pipeline_20261101/SUPERVISION_RESCOPE_V2.md
    revision: ccb8c44d211696bca02ea31645d0eca81f5b8ac27474fd3de4f9b57909fceecd
    locator: 真实独立Mask与合成物理可见标签的区别；旧任务目标仍保留原标注
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: BCNet、视频质量控制与生成不确定标签的范围限制
    relation: references
---

## 核心想法及联系

[中心纲要](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)中的“可靠实例分割”可以进一步问：哪些区域是图像支持的可见证据，哪些是身体补全或身份推断，哪些归属本就不确定？探索可见区域与推断层的区分、局部可信度，或少量合理实例关系候选。

不同应用未必需要相同输出：计数可能更重视对象是否存在，体尺可能需要更完整身体，局部行为可能更重视肢体归属。先理解需要什么证据，再决定是否值得做补全或不确定性建模；不将所有应用都扩成必须完成的新任务。

## 为什么值得思考

已有监督复核在有限真实样例中发现独立Mask重叠和设施填入，合成标签则来自物理可见性。它提示标签语义值得核实，尚不足以把全数据定性为amodal或宣布它是现有失败主因。

若模型被要求从同一画面给出超过证据支持的唯一答案，继续加语义头可能只增加看似确定的猜测。区分观测、推断和未知，可能使错误更可理解，并让[历史证据](IDEA_PIG_OBSERVATION_INFORMATION.md)、[人工询问](IDEA_PIG_RELATION_ANNOTATION.md)有明确使用位置。

## 可展开的可能性

一个方向是保留可靠可见实例片段，对争议边缘或归属另给局部可信度；另一个是把完整身体推断作为独立层，以免它与可见像素混淆。是否需要多候选或概率表达，应由具体用途决定，而非默认增加输出复杂度。

不确定性、amodal分割和分层遮挡都有直接前作。潜在价值需要来自猪场任务语义、可用证据与可靠决策的具体联系，尚无新颖性结论。

## 未知与边界

实际标签语义分布、不可辨对象比例，以及应用能否利用不确定输出仍未知。高置信也可能错误，给出置信度本身不能保证可靠。已有真实GT、旧评价口径和Study判定保持原义；新的任务定义只在未来独立设计中明确，不回改旧结果。

当前为候选，未开展新标注、评价或模型运行，也不要求先完成原概念主线。
