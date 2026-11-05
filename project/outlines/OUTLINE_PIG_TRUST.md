---
schema_version: 1
id: OUTLINE_PIG_TRUST
kind: outline
title: 什么样的输出才可信且有用？
parent_id: OUTLINE_PIG_INSTANCE_SEGMENTATION
outline_level: 2
summary: 连接可见证据、身体推断、未知、应用需求与评价，保留收益、损伤及泛化边界。
author: AI依据用户三层结构要求及既有研究整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 可见／amodal任务、视频质量控制、指标与输入权限
    relation: references
---

## 关键问题

可见像素、完整身体推断和不确定归属有不同证据基础。计数、体尺和行为分析也可能需要不同输出；可靠性要说明实际支持了什么、损伤了什么，以及应用是否能利用这些结果。

## 已有认识与排除范围

原真实独立Mask与合成物理可见Mask并不天然等价，有限样例不能给全数据重新定性。总体AP、辅助mIoU、GT候选机会、正常无GT效用与机制证据分别理解。旧真实GT与冻结评价保持原义，新问题不追改旧结论。

## 2026-11-04综合反馈

[评价缺口](OUTLINE_PIG_EVALUATION.md)和[综合复盘](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)补清：相对同条件普通续训的改善、相对保留强模型的替换价值，以及独立新条件可靠性，是不同结论。P优于R仍损伤N、片段／局部方法尚未测的事实，都要求保留收益与未知的具体范围，不能用一个“通过／失败”标签代替。

[本轮C21／V23消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)使选择保护的价值更清楚：强制完整候选提高修复数，却使原成功损伤由18增至244／178，并产生1376／1816条空Mask。当前更支持通过候选、回退和修正强度控制保护原有正确输出。共享部位保留图的小幅AP变化还伴随AP75与成功实例的不同方向变化，这些读数共同进入[评价缺口](OUTLINE_PIG_EVALUATION.md)和[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，作为保留原V23及选择简化候选的依据。

## 具体未解缺口

**2026-11-05形成的认识：** [夜间完整验收](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)显示，完整候选与风险策略共同重训后，一个seed仍出现226条编辑后空Mask及22／2126原成功损伤；形变置零／共享保留三seed均为0空Mask、护栏通过。多数空输出低置信，但少数与GT框高重叠，可靠性判断要把输出类型、实例损伤和总体AP一起解释。共享读出平均+0.148434的稳定小收益支持优先精简，当前实际算力收益尚待删除计算后验证。

同批C21保持项仅改变0／−1／+1个GT75成功状态，训练峰值显存由约3.13增至4.89GB而平均只+0.024948 AP；保留局部更新较稳定的认识，结束本版保持扩训。不同功能的采用价值继续沿[评价缺口](OUTLINE_PIG_EVALUATION.md)与[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)积累。

[可见证据、推断与未知怎样对应可信评价？](OUTLINE_PIG_EVALUATION.md)

## 从这里形成的Idea

[可见性与歧义表达](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)可以连接[历史观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)和[关系询问](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)。置信度本身不保证可靠，应围绕具体用途理解信息价值。

停止或保留一个Idea时，本页应留下判断范围与未决，并回到[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，而不是只保留一个通过／失败标签。
