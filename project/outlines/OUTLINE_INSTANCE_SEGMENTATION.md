---
schema_version: 1
id: OUTLINE_INSTANCE_SEGMENTATION
kind: outline
title: 实例分割的信息、表示与读出
parent_id: OUTLINE_PAPER_DISC
summary: 整理现有模型读出、对象知识与实例归属的联系，区分信息、学习和任务使用问题。
author: AI 整理，依据项目既有资料
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

## 要理解的问题

实例分割失败可能发生在目标提出、实例归组、局部边界读出、条件迁移和观察信息等不同环节。“分不好”本身不足以指定某个网络、原型库或部位监督方案。应先整理错误对象、可利用的信息和竞争解释，再形成可检验的 Idea。

## 已有表示的读出与修正

[原型—系数链路](../../paper/manuscripts/neighbor_sensitive_p3/README.md)研究固定模型中的可修机会、后训练兑现与损伤。[选择性掩码读出校准](../../paper/manuscripts/mask_boundary_calibration/README.md)研究一次修正是否有益，以及怎样减少对原成功对象的误伤。两条路径都需要区分可修上界与实际部署收益。

相关的[空间掩码读出稿件](../../paper/manuscripts/box_relative_mask_readout/README.md)也提供框相对位置、原生系数微调和空间读出的材料；它未独立登记为正式研究主线，保留为知识来源。

## 对象知识怎样获得并用于实例任务

猪群分割中的思路从跨图像原型字典，发展到可变形对象、具名部位、内容参与重建，再到训练期对应和概念监督。[Idea 演变轨迹](../research_notes/sources/idea_evolution_e93bf5358866376b.txt)说明，这些变化并非每一步都证明前一步无效，而是在不断区分：知识可被表达、知识能被学习，以及任务是否实际使用这些知识。

SPM 主线是这些想法的既有探索路径。其目录名称保留历史身份，具体研究范围需要结合所属版本资料理解，不能让最初方法名称替代当前研究问题。

## 猪实例分割的三层问题入口

[真实猪场的可靠逐头实例分割](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)按用户2026-11-03要求，组织为中心命题—五个关键研究问题—九个当前具体缺口。问题涉及实例组织、观察信息、监督价值、跨条件能力和可信输出；原数据、模型、训练、评价资料保留在相关缺口，跨问题Idea继续可关联多个纲要。该结构不限定模型、输入或学习范式。

[文献原文与阅读范围](OUTLINE_PIG_LITERATURE.md)、[既有探索与实验反馈](OUTLINE_PIG_EVIDENCE.md)和[跨问题候选联系](OUTLINE_PIG_HYPOTHESES.md)作为支撑资料保留。已有实验的阴性和暂停按各自对象解释；片段关系有限TRAIN实验结束后，条件可靠性新线已完成固定模型的成像干预，结果分别反馈到具体缺口，没有自动恢复其他线。

## 2026-11-04已有内容怎样转为受保护的修正

[猪分割C21／V23固定模型消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)显示，当前读出对内容输入、细尺度残差及外观码有明显依赖；强制完整候选增加修复，却以更多损伤和空Mask为代价。实例表示与是否应用修正需要共同理解，保护策略是已有正常收益的重要组成。下一问题聚焦保留内容总量时正确对应与选择器响应的分工，具体认识和研究安排已进入[猪分割中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。

## 2026-11-05信息价值与表示复杂度的进一步区分

[猪任务三组Idea完整训练](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)补充了两种不同认识：同COCO起点正确部位仍未稳定超非语义辅助，而已有V23任务保留读出加强共享约束后三seed均小幅改善。这使“额外标签的特异价值”和“已有读出的自由度是否必要”分别受到检验，当前优先保留功能精简候选，结束本版语义联合与局部保持扩训。正确内容怎样与选择保护协作、什么新信息值得获得，继续由[猪中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)及对应问题积累。

## Idea 的探索情况怎样回到这里

原有读出修正、选择性校准和猪知识Idea继续保留稳定身份。新的猪实例分割子纲要将开放候选与已执行版本分别关联；具体阴性、收益、暂停和限制仍沿原主线及实验查阅。新记录通过显式关联反馈认识，不因Run完成自动判定机制。
