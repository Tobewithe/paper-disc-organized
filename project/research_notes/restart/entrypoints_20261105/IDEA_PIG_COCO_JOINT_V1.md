# 入口整理前快照（2026-11-05）

原文件：[project/ideas/IDEA_PIG_COCO_JOINT_V1.md](../../../ideas/IDEA_PIG_COCO_JOINT_V1.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
---
schema_version: 1
id: IDEA_PIG_COCO_JOINT_V1
kind: idea
title: 从共同COCO起点联合学习猪部位知识与实例分割
summary: 三seed同COCO起点正式训练及独立验收完成；C−Q平均−0.513151 AP且种子异号，关闭本版语义联合训练扩训，保留一般辅助效应与长程退化认识。
outline_ids: [OUTLINE_PIG_INSTANCE_SEGMENTATION, OUTLINE_PIG_SUPERVISION, OUTLINE_PIG_TRAINING, OUTLINE_PIG_MODEL]
line_ids: [LINE_SPM_YOLOSEG_20261021]
exploration_status: closed
assessment: contradicted
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: A组9个训练配置、三个预定点、配对控制与原成功损伤
    relation: references
---

## 判断依据

[模型履历与整合审查](../../PIG_TRAINING_INTEGRATION_AUDIT_20261104.md)确认当前N经历多阶段选择。旧概念V1从N继续适配，C−Q平均+0.0882 AP且seed异号；原辅助梯度与原生参数更新成立。由此选择不同的学习阶段：从同一COCO资产构造单类模型，让正确部位在主要目标形成时共同参与。

普通A_R、非解剖空间分区A_Q、正确五部位A_C共用初始化、真实日程、合成RGB曝光、几何监督条件和主要预算。Q/C辅助结构与初值匹配。原N作为已有强模型参照，各方法都完整报告检测和Mask表现。C−Q区分正确部位与一般稠密分区，C−R评价辅助方案的整体作用；固定5／10／20轮曲线辨别收敛时序。

## 2026-11-05正式结果与判断

[夜间Study完整报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)及独立验收完成。每臂三个配对seed、2120更新均到末点；共同原生／辅助初值、日程、实际梯度、BN固定及fresh重载通过。A_R／A_Q／A_C平均Mask AP为52.515735／54.989314／54.476163；C−Q为−2.248916／+1.572748／−0.863283，平均−0.513151。C−R平均+1.960429，但Q−R更高，为+2.473579。

三个预定点的C−Q均差都为负。R／Q／C的5→20轮平均AP分别55.701→52.516、57.355→54.989、57.179→54.476，Box AP也随训练下降；正确部位末点损伤原N成功197／131／155例，均超过1%护栏。

本批更支持一般稠密辅助／正则化的相对作用，削弱了“把起点换成COCO即可得到稳定语义增量”的解释。知识获取、真实迁移和实例利用的分工仍待辨别。关闭本版联合训练的扩训与默认采用，保留学习曲线退化及普通辅助效应，反馈[学习缺口](../../../outlines/OUTLINE_PIG_TRAINING.md)、[监督问题](../../../outlines/OUTLINE_PIG_SUPERVISION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。原N+5目标与test426封存保留。

## 反馈循环后的复判条件

当前判断是**停止本版本、降低语义联合训练的优先级**，同时保留一般辅助有效和共同长程退化这两个局部认识。进入新版本的依据应是定位到可修的训练／信息瓶颈，或得到足以区别语义与普通辅助的新信息；随后用匹配控制检验该解释。正式采用时再要求预先保存的配对比较有稳定净收益且通过损伤护栏。单独的辅助loss下降、非零梯度、延长日程或更换起点缺少重开依据。

本次[汇总级诊断](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)显示9个A配置（3臂×3个配对seed）的Mask AP和输出量从5到20轮全部下降，输出约减少一半，中尺度损失明显；下一步先分解候选几何覆盖、排序保留和匹配Mask质量。原版本保持`closed／contradicted`，后继的执行范围和退出条件统一维护在报告第8.6节，逐例分析目前受远端SSH阻塞。
