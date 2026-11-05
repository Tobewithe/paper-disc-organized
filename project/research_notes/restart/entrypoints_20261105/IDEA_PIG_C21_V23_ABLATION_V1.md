# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_C21_V23_ABLATION_V1.md](../../../ideas/IDEA_PIG_C21_V23_ABLATION_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_C21_V23_ABLATION_V1
kind: idea
title: C21与V23的收益依赖哪些实际组件？
summary: 七项固定权重消融及独立验收完成；内容输入、细尺度、外观码及选择策略有明显功能作用，形变和分部位保留图作用较小。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL, OUTLINE_PIG_EVALUATION, OUTLINE_PIG_EVIDENCE]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: mixed
author: AI依据用户2026-11-04要求先做C21和V23消融形成，并按原实验及验收结果更新
curation_status: source_based_draft
evidence:
  - path: experiments/pig_c21_v23_ablation_20261104/RESULT_SUMMARY.json
    revision: 652dc5f6fa03f46f73ef0b45ad735020b0ee89b1d2f87358e8024e9512e43e31
    locator: 完整432dev、十组正常输出及七项功能干预
    relation: references
  - path: experiments/pig_c21_v23_ablation_20261104/ACCEPTANCE_RECEIPT.json
    revision: 042d65c77d9ffa3f5cc32a02389360ec9fb2e8ce4c68c83c9271e70eed8fa618
    locator: 1176独立完整性、COCO指标、GT状态及来源重抽复核
    relation: references
---

## 问题与事前依据

C21与V23相对项目保留N已有约3.07／3.41 AP收益，收益主导组件仍需分清。旧固定动作／质量的内容破坏较弱或覆盖不足，因而本次沿[协议V1](../../../../experiments/pig_c21_v23_ablation_20261104/PROTOCOL_V1.md)，在正常推理路径移除组件，并允许动作／风险选择重新计算。

工程1173完成2张预定图片、每臂54条输出；N／C21／V23原RLE重放、七项干预及状态恢复通过。原[启动交接](../../../../experiments/pig_c21_v23_ablation_20261104/HANDOFF.md)保留03:50:56 UTC核到1174运行8／432图、1175排队的快照。

## 实际结果

1174／1175完整推理与统计完成，1176独立验收通过，见[完整报告](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)。固定原checkpoint、432dev／2863GT、每臂12899条输出，0优化；N／C21／V23 Mask AP精确重现57.343500／60.416526／60.753278。

| 组件干预 | 相对完整模型Mask AP |
|---|---:|
| C21重建器内容输入置零 | −2.306973 |
| C21细尺度残差置零 | −2.565348 |
| C21 top1候选全强度应用 | −4.777370 |
| V23形变置零 | −0.028829 |
| V23外观码置零 | −1.315483 |
| V23部位保留图平均共享 | +0.075585 |
| V23完整候选直接应用 | −3.750892 |

两种强制完整应用将原N成功损伤由18增至244／178，并产生1376／1816条空Mask。形变和平均保留图也分别改变935／1124条Mask，实际干预已触发。

## 机制解释与方法判断

本轮判断为已结束／混合：当前模型明显依赖内容输入、细尺度重建、外观对应与选择保护；形变和逐部位保留图的AP作用较小。强制应用提高修复数却更大幅增加损伤，提示选择策略通过候选、回退与强度控制抑制错误或过强修正。去掉细尺度后修复由61减至20、损伤由18减至0，说明修复能力与保护需要共同考虑。

本轮采用参照保留原V23；优先研究内容重建、外观匹配与选择保护的分工，将形变和分部位保留列为简化候选。下一关键问题是：保持内容总量、沿正常动作重选时，正确空间／部位对应相对等量错配具有多大作用。该问题来自本次置零强效应与旧等量错配弱效应的差异。

本轮对象为固定权重功能依赖，使用既有四来源dev；总计0.607500GPUh／2.586667CPU-coreh，test426封存。完整证据位置见[存储定位](../../../../experiments/pig_c21_v23_ablation_20261104/REMOTE_STORAGE.json)。

## 对知识链路的反馈

具体认识进入[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)、[实例表示](../../../outlines/OUTLINE_PIG_MODEL.md)、[可信评价](../../../outlines/OUTLINE_PIG_EVALUATION.md)，再回到相关关键问题与[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)；[猪体知识Idea](../../../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)吸收内容使用的正面证据及正确对应的新问题。[SPM主线](../../../../paper/manuscripts/spm_yoloseg/README.md)据此调整优先级，当前安排沿[总控](../../../ACTIVE_PLAN.md)保存。
