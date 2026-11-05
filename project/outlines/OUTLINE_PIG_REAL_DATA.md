---
schema_version: 1
id: OUTLINE_PIG_REAL_DATA
kind: outline
title: 哪些少量标注能解除实例归属歧义？
parent_id: OUTLINE_PIG_SUPERVISION
outline_level: 3
summary: 比较完整Mask、归属关系与局部修正增加的信息和实际人工成本，保留数据覆盖及标注语义背景。
author: AI保留原知识材料并按用户三层结构重组
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构基准、猪框监督、视频研究及来源隔离条件
    relation: references
---

## 具体缺口与当前认识

已有独立Mask可能已包含所需同体／异体关系。少量关系或局部修正是否提供独立信息、对人是否更容易，尚未核实；由GT模拟标签数量不能证明人工成本下降。

[关系型标注Idea](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)与[片段组织](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)在这里交叉。[首个数据实验](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)已发现当前原Mask只提供14个可靠断连支持，未达预定量；结束直接依赖这一缓存训练显式断片的入口。剔除重叠区域的150个额外断连不能当作新增可见标签，也没有证明补标关系比完整Mask更有信息或更便宜。该认识同时反馈到[监督问题](OUTLINE_PIG_SUPERVISION.md)。

## 1. 数据集覆盖了怎样的猪与猪场？

比较品种、生长阶段、姿态、数量、密度、视角、昼夜、猪栏和农场来源。区分图像数量、独立个体数量与独立场景数量。

## 2. 标注提供了什么，又遗漏了什么？

比较实例Mask、框、关键点、姿态、身份和遮挡标签。分析可见区域与完整身体的标注差异，以及模糊、设施遮挡和身体接触处的标注歧义。

## 3. 标注成本是否限制了方法选择？

比较精细Mask、框、点、少量人工修正和视频辅助标注。研究在相同人工成本下，增加样本数量与提高标签精度各有什么价值。

## 4. 数据划分是否支持泛化结论？

检查随机帧、视频隔离、相机隔离和跨农场测试的差别。分析已有高分究竟反映同环境拟合，还是新条件下的能力。

**这里要回答：目前缺的是样本、条件覆盖、可靠标签，还是可信的独立评价？**

## 相关文献线索

[猪框监督研究](https://doi.org/10.1038/s41598-025-05113-x)提供不同目标域标签成本的比较；[MGB-Former](https://doi.org/10.3390/agriengineering8100447)明确按原视频划分。综述应分别核对其标注来源、预训练和测试权限，不由单篇数据集代表所有猪场。

[上层：监督信息价值](OUTLINE_PIG_SUPERVISION.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [相关缺口：受控合成](OUTLINE_PIG_DATA.md)
