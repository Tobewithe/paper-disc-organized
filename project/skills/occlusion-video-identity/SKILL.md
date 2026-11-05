---
name: occlusion-video-identity
description: >-
  Reason about visible versus amodal segmentation, physical occlusion, truncation,
  causal video memory and identity recovery. Use when defining occlusion labels,
  transferring synthetic video supervision or evaluating reappearance and tracking.
  遮挡、可见性、视频身份与评价知识。
metadata:
  version: "1.0.0"
---

# 遮挡、视频与身份

先确定要解释的是可见区域、完整投影、遮挡物、出画、身份还是最终跟踪结果。读取[定义、方法与反例](references/methods.md)；文献和数据集的实际阅读范围见[SOURCES.md](SOURCES.md)。

| 当前问题 | 需要区分的对象 |
|---|---|
| 遮挡标签怎样定义 | 可见Mask、画内完整投影、视野裁切、外部遮挡与自身不可见表面 |
| 遮挡后如何保持身份 | 过去可见证据、当前竞争者、时序因果性和拒绝错接 |
| 合成能力能否迁移 | 几何预测、表征干预、固定候选与完整系统效用 |
| 视频或基础模型是否公平可比 | 首帧提示、已知数量、未来帧、人工QC和检测召回 |
| 数据集能支持什么结论 | 点可见性、modal像素、amodal框与amodal像素分别核对 |

隐藏内容是预测，不能当作观测到的真实身份。完全不可见且没有历史的目标可能不可辨；不要求模型从无信息输入恢复唯一真值。判断收益时同时看正确恢复、错接、拒绝、覆盖与延迟。

本Skill整理既有阅读及项目实证边界，不激活原记录中的SAM适配、视频预训练或远端执行计划。
