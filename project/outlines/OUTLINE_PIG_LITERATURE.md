---
schema_version: 1
id: OUTLINE_PIG_LITERATURE
kind: outline
title: 资料：猪实例分割文献与阅读范围
parent_id: OUTLINE_INSTANCE_SEGMENTATION
summary: 三层研究问题的支撑文献资料；保留实际原文核读、检索完成度和证据边界。
author: AI整理，依据实际检索与指定正文核读
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 16篇指定正文证据卡及未完成近邻
    relation: references
---

## 与问题纲要的关系

本页作为[真实猪场实例分割中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)的参考资料保留。文献证据分别进入相关问题与具体缺口，支持形成Idea，不作为必做章节，也不把检索和下载数量当研究发现。原阅读范围与近邻缺口不因结构整理而改变。

按问题查找后续知识积累见[知识入口](../research_notes/README.md)，其中[多任务Skill的辅助学习正文](../skills/multi-task-optimization/references/auxiliary-transfer.md)连接2026-11-05的阅读。本页下方的检索数量与逐篇分析继续对应2026-11-02那次记录。

## 直接猪研究提供什么认识

| 工作 | 可借用的证据 | 不能越过的边界 |
|---|---|---|
| 多架构猪分割基准 | 不同表示都可工作，遮挡归属和细部优劣仍不同 | 不能由少数旧配置断言规模永远无效 |
| CClusnet-Inseg | 中心投票有助于把遮挡后的可见片段归回实例 | 有额外中心标签、随机帧拆分和聚类代价 |
| 猪框监督分割 | 真实框加教师伪Mask可接近同篇部分全监督模型 | 使用预训练；单猪场结果不等于跨场确认 |
| MGB-Former | 普通实例Mask引导在其消融中贡献明显 | 有MP-Former直接先例，模块容量不同 |
| STARFORMER与猪视频基础模型 | 多任务或时序观察可以参与监控 | 已知数量、参考帧、QC和离线信息要单列 |

原文链接、指定阅读范围和论文自报结果见[逐篇证据卡](../research_notes/literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md)。不同论文的AP、F1、PQ和J&F不排列成统一榜单。

## 相关方法怎样进入分析

- **实例表示与任务组织**：Mask2Former、MP-Former、BCNet、TAPPS分别提供集合表示、匹配稳定、遮挡分层、对象—部位绑定的思路。
- **数据与迁移**：Copy-Paste、合成猪检测、SAP、Dataset Diffusion、DAFormer分别提供重组、结构标签、生成伪标签和域适配的证据。
- **阅读顺序**：先核问题和权限，再读机制与对照，最后判断哪些条件能转移到猪场。不能只凭模块名相似推断适用性。

## 检索完成度

本轮1137条原始记录去重为961条，含不相关条目，未完成逐条筛选。公开获取74条记录中的61份PDF及另外7篇XML全文；实际核读16篇指定方法/实验段，不是16篇逐页全文，也不是全部下载论文已读。

S2、DBLP和部分网页阻断保留。PigInstance、部分猪仔amodal研究尚缺可读全文，因此新颖性审查仍有明确缺口。完整查询、来源、失败、SHA和阅读范围由[检索状态](../research_notes/literature/pig-instance-open-analysis_2026-11-02/SEARCH_AND_READING_STATUS.json)及[阅读日志](../research_notes/literature/pig-instance-open-analysis_2026-11-02/READING_LOG.json)追溯。

## 文献卡的固定分析问题

每篇回答：实际任务和信息权限是什么；错误有何证据；改变了哪个计算过程；数据、容量和预算是否匹配；结果排除了哪些解释；还有哪些解释未排除；对新方案有什么可测的启发。下载、摘要、指定段落和全文阅读分别标记。
