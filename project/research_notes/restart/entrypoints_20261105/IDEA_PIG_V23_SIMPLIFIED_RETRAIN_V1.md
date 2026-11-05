# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md](../../../ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1
kind: idea
title: 对V23低依赖组件施加约束后重新学习候选与选择
summary: 三seed候选／成本／风险重训和独立验收完成；形变置零平均+0.010234、共享保留+0.148434 AP，支持功能简化，优先保留共享版本用于后续实际压缩。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRAINING, OUTLINE_PIG_EVALUATION]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: supported
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: B组9配置、简化配对差、护栏、空Mask与实际耗时
    relation: references
---

## 判断依据

[已验收消融](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)中，V23形变置零−0.028829 AP、九部位保留平均共享+0.075585 AP，而外观码与完整候选分别−1.315483／−3.750892。前两项提供了优先简化的依据，也留下重新训练后系统怎样调整的问题。

B_FULL完整控制、B_NO_FLOW固定形变输出为0并冻结该分支、B_SHARED_VIS在正常读出中平均九部位保留量，均从同一已训练V23开始。每组独立完成4000候选更新、完整1592 TRAIN图的实际640二值动作成本和8epoch风险训练。正常动作、训练标签定义和原N／C21参照保持一致。

本批是功能约束版本；报告实际训练参数和耗时。满足配对平均AP差≥−0.1且护栏稳定的配置，进入后续结构压缩候选。若新完整控制和简化臂共同退化，则结合原V23判断共同适配造成的变化，优先保留已有模型。

## 2026-11-05正式结果与判断

[完整报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)及独立验收完成。九个配置均4000候选更新，九份1592图／41059行成本完整，风险各8epoch／2568更新。完整／形变置零／共享保留的平均Mask AP为60.668213／60.678446／60.816647；相对完整控制分别+0.010234／+0.148434。共享三seed差+0.192220／+0.065603／+0.187479，方向一致；两种简化均通过原非劣筛选及全部采用护栏。

完整控制首seed出现226条编辑后空Mask，原成功损伤22／2126略超1%；简化版本三seed均0空Mask。空输出多数低置信，仍有少数与GT框高重叠。共享版本平均仅比原V23高0.063368 AP；四来源重抽在两个seed跨0，当前作用量支持精简候选筛选。

与先前固定权重消融共同看，结果支持限制任务保留读出的自由度，候选及风险策略可在约束下共同补偿。优先保留共享保留图，形变置零次之，原V23继续作成熟参照。本批均值／置零实现尚执行原卷积，推理耗时没有稳定下降；下一问题是实际删除计算、核读出等价性与效率，而非继续放大完整结构。认识进入[表示缺口](../../../outlines/OUTLINE_PIG_MODEL.md)、[可信评价](../../../outlines/OUTLINE_PIG_EVALUATION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。本V1验证结束，结构压缩尚未执行。

## 反馈循环后的复判条件

当前判断是**保留共享保留图作为优先功能精简候选、形变置零作为次选，停止以完整结构再训换取明显涨点**。下一步先测目标分支在端到端成本中的占比，存在实际回报才进入算子删除和等价性／近似误差检验；压缩后的精度、空Mask、原成功损伤和预热后端到端时延共同决定是否采用。对应版本若有关键护栏失败，或成本收益达不到本轮目标，则停止该压缩对象并保留原V23。本轮新保存的成本目标、有限预算和退出条件统一见[报告第8.6节](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)，原功能简化的`closed／supported`判断保留；真实成本测量尚受远端SSH阻塞。
