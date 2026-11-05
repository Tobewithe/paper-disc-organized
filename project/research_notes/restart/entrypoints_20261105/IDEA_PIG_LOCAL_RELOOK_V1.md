# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_LOCAL_RELOOK_V1.md](../../../ideas/IDEA_PIG_LOCAL_RELOOK_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_LOCAL_RELOOK_V1
kind: idea
title: 本帧局部重看修复框较准的Mask失败：V1
summary: 固定项目保留N与原128TRAIN病例池，合格失败仅15例、未达预定32；当前入口关闭、局部重看方法未测。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_OBSERVATION, OUTLINE_PIG_DETAIL_EVIDENCE, OUTLINE_PIG_TASK, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_PIG_ADAPTIVE_DETAIL_20261103]
exploration_status: closed
assessment: insufficient
author: AI依据用户2026-11-03接受推荐并开展最小验证的指令
curation_status: source_based_draft
---

## 具体验证主张

在同条正常检测框IoU≥0.75、MaskIoU处于0.50–0.75的实例上，用原预测框生成局部重看，可利用本帧已有像素改善Mask；同尺度低细节对照用于限制纯尺度／上下文解释。这是[广义按需细节Idea](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)的一个具体版本，不把插值当作新观察，不继承旧固定池质量头或部位方案。

按[协议V1](../../../../experiments/pig_local_relook_20261103/PROTOCOL_V1.md)只在已有128 TRAIN与原预测中固定32失败＋32成功，任一支持不足即结束当前池入口。继续线为细节组修复≥8、较低细节组多修复≥4、原成功新增失败0，均以32为分母；GT不进入裁剪与新输出挑选。该筛选线不等于AP、总体1%保护或独立泛化。

## 当前执行与边界

用户已明确授权本最小验证，全程在登记服务器执行；本地只保存总结和来源指针。Slurm1160完成资格：128图／1460GT、2261正常输出、1456个同条框75关联；其中仅15个合格失败、1441成功，不足32失败。1161用独立关联及位运算重算全部1456项，误差0。两作业正常完成，共6CPU-core秒、0GPU／新模型前向／优化。

具体入口按原协议结束，记已关闭／支持不足，**方法未测**。没有降低阈值、改用15分母、追加图像或从另一检测拼接成功。完整结论见[报告](../../../../experiments/pig_local_relook_20261103/REPORT_V1.md)。用户随后要求夜间较长训练序列，另立[训练Idea](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)，不重开本入口，不追改原Gate。广义按需细节继续未知。

## 2026-11-04综合回顾的边界

[近期总结](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md#6-三个首试没有方法成绩的地方保留未测)复核本条仍为`closed/insufficient`、方法未测；受测N是多阶段项目保留权重，不能从已见TRAIN病例少推出目标任务初始模型或完整开发集无需局部观察。夜间裁剪训练属于另一个Study，其阴性不补作本试验未执行的局部推理结果；本边界回到[广义细节Idea](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)、[细节缺口](../../../outlines/OUTLINE_PIG_DETAIL_EVIDENCE.md)和[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，当前池不重开。

数据、代码、逐例输出和冻结版本位于服务器`/srv/gpu-lab/users/a25023104/projects/paper-disc-organized/experiments/pig_local_relook_20261103`。执行路由见[本线入口](../../../../paper/manuscripts/pig_adaptive_detail/README.md)，原始科学状态以真实Run与后续报告为准。
