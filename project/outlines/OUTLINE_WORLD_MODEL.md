---
schema_version: 1
id: OUTLINE_WORLD_MODEL
kind: outline
title: 世界模型的行动时间可辨识性
parent_id: OUTLINE_PAPER_DISC
summary: 整理端点预测、动力学解释与行动判断的区别，关注什么观察能够区分相容模型。
author: AI 整理，依据项目既有资料
curation_status: source_based_draft
evidence:
  - path: project/knowledge_sources/world_model_action_time_identifiability_236755767994e3b1.txt
    locator: 原文 paper/manuscripts/world_model_action_time_identifiability/README.md
    revision: 236755767994e3b1610c96a64587fb84284cc947f19fc971875b4e4f2f16f338
    relation: references
---

## 观察与行动判断之间的距离

一个模型可以准确预测固定采样间隔下的观测端点，但这是否足以支持对区间内切换动作的排序，仍是需要研究的问题。应区分数据拟合、动力学可辨识性，以及新控制协议下的真实决策能力。

## 已有研究问题

[所属主线](../../paper/manuscripts/world_model_action_time_identifiability/README.md)提出寻找与现有数据相容、却对某个动作给出不同判断的模型，再通过有限、受控的区间内动作获取区分证据。解析构造、神经模型表现与主动探测效率分别需要自己的验证。

## 尚待建立的认识

随机种子集成的一致性是否能够识别结构歧义、什么探测最有价值，以及区分证据能否改善真实决策，均不能从主线登记或设计草案推出。该主题独立于视觉实例分割，各自保留任务定义与证据边界。

本纲要可以继续积累相关知识与文献。当前关联的 Idea 只是将已有研究问题接入知识结构，不认定任何未核实实验已经发生。
