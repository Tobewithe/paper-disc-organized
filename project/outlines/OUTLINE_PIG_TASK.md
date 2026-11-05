---
schema_version: 1
id: OUTLINE_PIG_TASK
kind: outline
title: 哪些真实错误属于实例组织问题？
parent_id: OUTLINE_PIG_ORGANIZATION
outline_level: 3
summary: 用可追溯对象和分母区分目标遗漏、片段归组、异体混淆与边界误差，为Idea选择提供依据。
author: AI保留原知识材料并按用户三层结构重组
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 猪领域基准、遮挡分类和不同任务条件
    relation: references
---

## 具体缺口与当前认识

本项目尚未可靠量化哪些正常错误真正涉及归组。标注断连、训练缓存分辨率和物理可见断片需要分开，文献错例也不能代替目标数据分母。

[片段组织Idea](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)由此产生；[片段关系主线](../../paper/manuscripts/pig_fragment_relation/README.md)的首试已完整核1272图／14642原Mask，仅14个可靠断连支持，停止当前缓存入口。见[实验报告](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。这个标签统计没有测正常模型错误，不能据此排除实例归属瓶颈；原部位版本失败也不代表归组被整体否定。结果同时反馈到[实例组织问题](OUTLINE_PIG_ORGANIZATION.md)。

## 支撑知识：任务、对象与环境

这一部分先定义研究对象，辨别“针对猪”究竟体现在哪里。

## 1. 任务目标与应用需求

讨论猪的实例Mask分别用于个体计数、行为分析、体尺体重估计、健康与福利监测时，需要怎样的精度。不同应用对身体完整性、细部、边界、身份连续性和实时性的要求并不相同。

区分单帧可见实例分割、完整身体估计和视频实例分割，说明它们各自需要什么输入、标签与评价。

## 2. 猪本身带来的困难

分析同类个体外观相似、身体柔性形变、站卧姿态变化、体型与生长阶段差异，以及腿、耳、尾等细部的像素尺度。这些特点分别影响目标发现、实例区分还是局部轮廓，不能笼统归为“猪难分”。

## 3. 猪场环境带来的困难

分析密集接触、身体重叠、栏杆与设施遮挡、昼夜成像、污渍、模糊、相机视角等条件。区分猪特有问题与其他密集动物、行人或细胞分割共有的问题。

## 4. 从现象形成错误分类

将文献中的失败归纳为目标遗漏、邻猪合并、同体拆分、局部归错、边界偏差和跨场景退化，比较哪些有统计证据，哪些仅来自示例或作者动机。

**这里要回答：什么问题值得专门研究，什么问题可能由现有通用方法解决？**

## 相关文献线索

[猪仔中心聚类研究](https://arxiv.org/abs/2206.01942)区分身体被分隔和末端缺失两类遮挡；[多架构猪基准](https://doi.org/10.52825/bis.v1i.59)比较了不同模型的局部归属错误。它们为分析提供线索，不能替代目标场景中的失败分母。

[上层：实例组织问题](OUTLINE_PIG_ORGANIZATION.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [相关缺口：标注的信息价值](OUTLINE_PIG_REAL_DATA.md)
