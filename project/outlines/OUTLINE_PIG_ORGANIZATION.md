---
schema_version: 1
id: OUTLINE_PIG_ORGANIZATION
kind: outline
title: 怎样把视觉证据组织成每一头猪？
parent_id: OUTLINE_PIG_INSTANCE_SEGMENTATION
outline_level: 2
summary: 区分目标发现、同体归组、异体分离和边界问题，比较实例表示所需的信息。
author: AI依据用户三层结构要求及既有研究整理
curation_status: source_based_draft
evidence:
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 当前缓存的原断片支持量及重叠剔除诱发断连
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构猪基准、CClusnet、Mask引导与对象部位绑定
    relation: references
---

## 关键问题

一头猪的可见证据可能分散，相邻猪的外观又可能相似。目标发现、同体连接、异体区分与局部边界是不同问题；先理解哪种错误真实存在，再决定是否改变表示、关系或训练组织。

## 已有认识与排除范围

文献对猪仔断片归组有正面相关依据，但本项目尚缺可靠的目标错误分母。V32的竞争相对小收益与部位错位近乎无损，不支持把普通竞争直接解释成具名语义作用。[固定池质量头V25／V26](../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)已停止，不能反复以同一GT池机会为依据重开。

[片段关系首试](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)进一步发现：当前TRAIN原160网格Mask的可靠断连仅14／14642实例，未达最低支持量，结束这一缓存入口；剔除重叠后出现的150个额外断连不能改作物理可见样本。原标注连通不等于图像证据连通，真实归属错误仍未量化。

[本轮C21／V23固定模型消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)补充了实例读出的实际依赖：内容输入、细尺度残差及外观码分别明显影响正常AP，强制完整候选则造成大量损伤与空Mask。当前更值得理解内容组织与选择保护怎样协作；下一判别聚焦保持内容总量时正确空间／部位对应的作用。形变与逐部位保留图进入简化候选，已有V23保留。具体认识保存在[表示缺口](OUTLINE_PIG_MODEL.md)，并回到[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。

## 具体未解缺口

**2026-11-05反馈：** [功能约束后的实际重训](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)保留了V23内容重建的整体收益，共享任务保留图比完整重训三seed均提高，平均+0.148434 AP；形变置零近乎持平，两者均过护栏。这使表示复杂度与正常选择的协同更具体：具名部位字段可以继续承载内容，任务保留读出则可采用更强共享约束。原生五部位辅助在同COCO起点仍未稳定超非语义控制，具名语义的特异价值和一般实例组织价值继续分别研究，认识回到[表示缺口](OUTLINE_PIG_MODEL.md)与中心纲要。

- [哪些真实错误属于实例组织问题？](OUTLINE_PIG_TASK.md)
- [不连通的可见片段怎样属于同一头猪？](OUTLINE_PIG_MODEL.md)

## 从这里形成的Idea

[片段表示与信息利用](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)研究表示本身，[关系型标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)研究怎样提供所需关系。它们可以结合，也要面对已有Mask已含足够信息这一竞争解释。需要新观察时连接[观察问题](OUTLINE_PIG_OBSERVATION.md)，不将单帧重组设为唯一答案。

实验结果应回写具体缺口与本页：到底排除了某个实现、改变了瓶颈判断，还是仍缺证据；再据此修正Idea。
