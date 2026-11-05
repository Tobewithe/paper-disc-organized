---
schema_version: 1
id: IDEA_PIG_LOCAL_RELOOK_V1
kind: idea
title: 本帧局部重看修复框较准的Mask失败：V1
summary: 原病例池未满足预定失败支持量，入口关闭、局部推理方法未测；后续夜间训练属于独立检验。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_ADAPTIVE_DETAIL_20261103]
exploration_status: closed
assessment: insufficient
author: AI依据用户2026-11-03接受推荐并开展最小验证的指令
curation_status: source_based_draft
---

## 本版主张与检验

对同条正常检测中框较准、Mask处于中间质量区间的实例，用预测框生成局部重看，检验本帧已有像素能否改善Mask；同几何低细节对照用于区分细节和尺度／上下文效应。[协议](../../experiments/pig_local_relook_20261103/PROTOCOL_V1.md)固定原病例池、失败与成功支持量及继续线，GT仅用于诊断分组和评价。

## 当前判断与依据

原池资格不足，按协议关闭入口，局部推理方法未测。独立复核确认失败支持量未达要求，未扩大样本池或改分母，完整证据见[报告](../../experiments/pig_local_relook_20261103/REPORT_V1.md)。

受测N为多阶段项目保留权重，已见TRAIN上的病例少仅限制本次池入口；完整开发集的局部观察价值仍未知。[夜间裁剪训练](IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)由随后独立授权的Study检验，其结果沿自身范围解释。[广义按需细节](IDEA_PIG_ADAPTIVE_DETAIL.md)继续保留其他观察与输入途径。

若重新研究，需要新的合格病例或信息依据，按新的实际对象设计比较。当前行动见[当前安排](../ACTIVE_PLAN.md)，运行与定位见[Study](../../experiments/pig_local_relook_20261103/study.json)。[整理前全文](../research_notes/restart/entrypoints_20261105/IDEA_PIG_LOCAL_RELOOK_V1.md)保留原分母、作业及复核细节。
