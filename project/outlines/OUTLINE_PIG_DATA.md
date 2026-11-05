---
schema_version: 1
id: OUTLINE_PIG_DATA
kind: outline
title: 受控合成能提供哪些缺失信息与可区分证据？
parent_id: OUTLINE_PIG_SUPERVISION
outline_level: 3
summary: 比较合成作为训练语料、额外监督和受控认识工具的不同价值，连接覆盖、成本、模型依赖及真实迁移。
author: AI保留合成知识并按用户三层问题结构整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: Copy-Paste、合成猪检测、SAP与Dataset Diffusion
    relation: references
---

## 具体缺口与当前认识

合成可以供训练、额外监督或受控认识。有限release证明标签和工程能力，不证明覆盖真实猪场；哪一种受控变化能帮助理解实际错误、是否超出普通增强，仍需要具体依据。

[受控合成Idea](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)和[覆盖迁移](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)从不同用途切入，联系[监督问题](OUTLINE_PIG_SUPERVISION.md)与[泛化问题](OUTLINE_PIG_GENERALIZATION.md)，不把写实度或合成数量当默认目标。

## 1. 为什么需要合成？

分别分析扩大外观变化、补足稀有姿态、构造密集遮挡、降低标注成本、获得额外几何标签和建立受控对照等用途。每一种用途需要不同的生成能力。

## 2. 不同合成方式的能力边界

比较二维复制粘贴、图像合成、程序化三维渲染、生成模型与图像编辑。重点分析它们能够改变哪些因素，能否提供可靠实例身份，以及制作和验证成本。

## 3. 应当合成什么？

从实际问题推导需要覆盖的个体差异、姿态、猪群关系、设施、相机与成像条件。讨论单因素变化与联合条件覆盖的差别，以及数量增加是否真正带来新信息。

## 4. 合成图像与标签如何保持一致？

分析遮挡后的Mask更新、附件归属、细部可见性、生成编辑造成的几何变化，以及不确定标签的处理。讨论额外部位、深度、表面和对应标签是否值得提供。

## 5. 怎样判断合成数据有用？

比较视觉真实性、标签质量、条件覆盖和真实任务收益之间的关系。分析合成数据在何种真实标注规模、训练预算和目标域条件下有效，何时反而损伤性能。

**这里要回答：合成的独特价值是什么，它是否超过更便宜的真实标注或普通增强？**

## 从现有语料到可检验的信息价值

[90图辅助release](../../experiments/spm_concept_synthesis_repair_20261101/REPORT.md)已完成身份/标签资格；30个几何组的A/B/C不是90个独立场景，身体与环境家族仍各一个。它证明有有限可用监督，不能证明代表真实猪场或改善AP。真实原独立Mask与合成物理可见Mask的语义差异仍须保留。

同一可控能力可以服务不同想法：补充实际缺失条件、提供难获得的监督，或成为[受控认识工具](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)，通过少量成对变化理解模型怎样依赖场景与实例关系。后者不要求最终方法使用合成训练，也可能为标注、表示或额外观察产生更好的Idea。

此前`C/Q × 原覆盖/补缺覆盖`只回答覆盖是否调节语义价值，是[覆盖与迁移](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)的一个局部候选，不能定义合成研究的全部用途。真实概念检查也不是所有生成、增强或认识工具构思的前提。

已有Study的生成输入、GT和拆分按各自合同保持；不将未验证属性当场景真值，也不因打开思路自动新增生成或训练。新Idea先说明增加了什么信息、认识或成本价值，实际选择时再细化适用的数据与比较。

## 已完成的受控认识首试

[成像条件实验](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)实际复用了全部30对A/B，核相同body、相机、几何／成像映射签名和标签数组；固定原生模型下，辅助owner有效域的实例成功数均278，配对差异为0。与128真实TRAIN色调操作的0.136986%最坏损失分别解释，两侧均未支持预定大幅粗粒度能力损失。

它说明受控合成可以用来限制一个具体解释，不需要靠扩大训练或追阳性证明工具价值。合成标签仍只作辅助诊断，不造正常分割GT；原图IoU50触顶、已见真实来源及单一家族限制外推。接下来需要实际新条件、细边界或置信度证据，不能从本阴性跳到“所有光照都无影响”或自动训练保持模块。

## 相关文献线索

[Copy-Paste](https://arxiv.org/abs/2012.07177)提供低成本重组参照；[SAP](https://arxiv.org/abs/2311.18661)涉及合成结构标签与真实部位迁移；[Dataset Diffusion](https://arxiv.org/abs/2309.14303)涉及生成图像与不确定语义伪标签。不同任务和标签权限应分别解释。

[上层：监督信息价值](OUTLINE_PIG_SUPERVISION.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [相关缺口：覆盖与迁移](OUTLINE_PIG_TRANSFER_GAP.md)
