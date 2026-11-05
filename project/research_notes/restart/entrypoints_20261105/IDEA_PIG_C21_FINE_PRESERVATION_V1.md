# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md](../../../ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_C21_FINE_PRESERVATION_V1
kind: idea
title: 限制C21细尺度适配并保持已学会的输出
summary: 六个局部5201参数配置及独立验收完成；保持项平均仅+0.024948 AP且种子异号，GT75变化0／−1／+1，结束本版扩训，保留局部更新较稳定的认识。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL, OUTLINE_PIG_TRUST]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: mixed
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: C组末点、保持配对差、原成功损伤及GT75变化
    relation: references
---

## 判断依据

[消融报告](../../../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)中C21去细尺度后−2.565348 AP，修复从61降为20，原成功损伤从18降为0，说明细尺度承担重要修复并伴随局部损伤。[旧夜间结果](../../../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)中的保持项相对无保持有效，但全非BN适配后仍低于强N；旧配置已关闭。

新版本把可塑性限制到C21的fine_encoder与fine_refiner，其余主体、内容、原生定位及原策略固定。C_FINE用真实task BCE／Dice／边界监督；C_KEEP在相同范围和预算下，加原C21同ROI／同候选的Bernoulli输出KL。保持区域为教师概率距TRAIN目标小于0.1的已知像素，教师固定。推理仍正常重算原策略，真实GT只进入训练损失。

这里检验的是限制参数范围之后，输出保持能否改善修复／损伤取舍。若KEEP损伤更低但修复也同步消失，需要据完整效用判断约束强度与候选机会；若两臂都无增量，则降低本次细尺度再适配配置的优先级，保留原C21／V23与细尺度功能认识。

## 2026-11-05正式结果与判断

[完整报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)及独立验收完成。六个配置均4000更新，5201参数白名单、教师与其余状态保持、KL生效及最终权重fresh重载通过。FINE／KEEP平均AP为60.318688／60.343635，相对原C21分别−0.097839／−0.072891；全部护栏通过且0空Mask。

KEEP−FINE为+0.036212／−0.003199／+0.041831，平均+0.024948；三seed GT75成功集合分别完全相同、损失1例、恢复1例。FINE原N成功损伤16／16／17，KEEP均16；修复分别60／61／60与60／60／60。末期KL约1.74–2.09×10^-4，保持项确已生效，但在本局部更新范围中改变的最终决策很少。教师前向使训练峰值显存由约3.13增至4.89GB，部署结构相同。

关闭本版细尺度再适配及保持扩训，保留原C21／V23以及“局部更新较稳定、KL边际作用很小”的认识。旧大范围保持的起点和任务另有条件，继续按原配对结果保存；本批降低的是在现有细尺度上继续加保持项获取新增AP的优先级。认识反馈[学习](../../../outlines/OUTLINE_PIG_TRAINING.md)、[适配](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)、[可信性](../../../outlines/OUTLINE_PIG_TRUST.md)与[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。

## 反馈循环后的复判条件

当前判断是**停止本版本局部再适配和继续加KL的扩训**，保留局部更新较稳定、保持项已生效但边际很小的认识。只有新的起点、参数范围或额外信息显示明确的可修复机会，才进入有匹配控制的新试验；拟采用时需要稳定改善GT75净修复／损伤取舍，并抵偿额外训练成本。如果新增观测显示KEEP与FINE的输出、成功集合和损伤仍基本相同，则结束该保持对象。当前版本继续`closed／mixed`，原证据不因新问题被清零；后续安排见[反馈循环](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。
