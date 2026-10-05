# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Project Direction
# 研究简报：密集猪群实例分割中的实例碎片化与候选竞争

> 建立日期：2026-09-04。本文档将 `C:\Dpan\codexproject\pigcv_research` 的既有分析接入当前 ARIS 项目；原始结果、代码和数据不复制、不改写，仍以原路径为准。

## 问题陈述

密集接触猪群中的实例分割不仅会出现传统的高重叠重复预测，还可能出现同一真实实例被多个低重叠局部预测分别解释的实例碎片化。现有离线失败分析显示，普通 Mask-NMS 能处理大量高重叠重复，但不能直接处理低重叠、互补的局部实例假设，以及完整候选与局部候选之间的错误排序或共同存活。

本研究的主线是一个“诊断到改进”的综合任务：先系统探明密集场景中实例分割失败率高的深层机制，再针对被证实的主导机制设计可实现、可证伪的改进，并检验其是否同时改善整体分割指标与关系型失败。候选竞争、遮挡关系、层/集合结构、中心聚类等都只是待比较的机制族；当前没有选定方法或论文主贡献。

## 背景

- **领域**：计算机视觉、智能养殖、实例分割。
- **子方向**：密集/接触目标实例分割、实例碎片化、候选或 query 竞争、实例关系建模、遮挡结构与评测诊断。
- **文献状态**：Research Wiki 当前有 27 个论文页面；核验状态以各页和 literature registry 为准，检索记录包含重复与
## Open Gaps
# Gap Map

更新时间：`2026-09-07`。本文件只表达证据支持的缺口与未成立推断，不选择论文主贡献。

最新续接：R005c 五条件计算和统计已完成，统一 .01 保留恢复 12、退化 269；GT 联合恢复 43 个所选目标及 2 个邻居。下一步不再重复同一阈值实验。用户选择优先研究 Faro 单 mask 质量；R005d 将 307 个 raw 严格质量缺失失败分为框支持不足 37、支持可行但 mask 缺失 270。原始标签到 COCO 的栅格转换一致；modal/amodal 政策仍待原文核验。这些结果正在补齐独立审查，不接受 scorer 路线。最新报告为 `yolo26_confidence_diagnostic_20260907.md` 和 `faro_mask_quality_20260907.md`；下表中 2026-09-05 的“下一步”保留为历史决策依据。

| Gap ID | 文献已覆盖机制 | 项目观测 | 尚未成立的推断 | 可审查下一步 |
|---|---|---|---|---|
| G1-low-iou-fragment | Layering、BCNet、ORM、OCFusion 主要处理密集重叠、遮挡边界或遮挡顺序 | `SPLIT_TYPE=50` 中联合 IoU 增益中位数 0.084、coverage 增益 0.143；31/50 完整与局部候选共同存活 | 低 IoU 互补碎片是否构成独立机制类别，文献没有直接证明 | 检索 low-IoU fragmentation、partial mask candidates、instance split error 的原始论文；对候选对做关系标注 |
| G2-candidate-selection | Mask Scoring R-CNN 已提出 mask 质量评分；已核验的 Topology-Aware Query Selection 使用 frozen Mask2Former 候选图、基数预测和精确子集选择 | 严格 coverage/purity .75 raw 可用性为 PigLife 360/394、Faro 490/797；final 可用/匹配为 312/310 和 224/224 | 通用关系选择已有直接近邻；GT 引导恢复不能证明排序因果或部署收益；固定 conf=.05 单独扩大 Top-K 无效 | 从同一基线分别检验置信度保留、删除及联合操作，并保留整体 AP 与全部回退 |
| G3-pig-domain-transfer | CClusnet-Inseg 用中心+DBSCAN；Pig Panoptic 提供猪群任务基线；MaskDis/Enhanced Mask R-CNN 为领域先例 | 旧诊断显示 36/50 `SP
## Key Papers (18 total)
- [paper:baselizadeh2025_occlusionordered_semantic_instance] Occlusion-Ordered Semantic Instance Segmentation: OOSIS jointly predicts instance masks and relative occlusion order from oriented occlusion boundaries and semantic segmentation.
- [paper:bolya2020_tide_general_toolbox] TIDE: A General Toolbox for Identifying Object Detection Errors: TIDE measures output-error contributions using separate GT-informed AP oracles relative to the same baseline, avoiding order-dependent progressive err
- [paper:compte2025_housed_pig_identification] Housed pig identification and tracking for precision livestock farming: Introduces the FaroPigSeg and FaroPigReID-33 datasets within a pig identification and tracking pipeline; retrieved official documentation leaves segme
- [paper:großkopf2026_redefining_instance_matching] Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation: The paper reformulates panoptic instance matching as constrained bipartite assignment and compares one-to-one, many-to-one, one-to-many, and many-to-m
- [paper:haidar2024_maskuno_switchsplit_block] MaskUno: Switch-Split Block For Enhancing Instance Segmentation: MaskUno routes refined ROIs through a Switch-Split block and specialized mask predictors to reduce competing mask kernels.
- [paper:he2025_segment_concealed_objects] Segment Concealed Objects with Incomplete Supervision: SEE combines mean-teacher pseudo-labeling with hybrid-granularity feature grouping for concealed-object segmentation under incomplete supervision.
- [paper:henrich2025_benchmarking_pig_detection] Benchmarking pig detection and tracking under diverse and challenging conditions: PigDetect and PigTrack benchmark individual pig localization and tracking under realistic barn conditions, i
## Recent Relationships (10 total)
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-stage-availability-fulltrace-20260905
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-r006-fixed-fulltrace-20260905
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-gate-binding-20260905
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-strict-quality-20260905
  idea:diagnosis-to-improvement --inspired_by--> paper:prytula2026_qcell_recombining_aligning
  idea:diagnosis-to-improvement --inspired_by--> paper:zhang2026_topologyaware_query_selection
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-confidence-20260907
  idea:diagnosis-to-improvement --tested_by--> exp:faro-box-support-20260907
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-external-full-20260907
  idea:diagnosis-to-improvement --tested_by--> exp:yolo26-o-criterion-review-20260907
