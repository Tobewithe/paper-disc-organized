---
schema_version: 1
id: OUTLINE_PIG_MODEL
kind: outline
title: 不连通的可见片段怎样属于同一头猪？
parent_id: OUTLINE_PIG_ORGANIZATION
outline_level: 3
summary: 比较实例表示及结构信息的使用，结合V23、V32与原生概念实验区分表示存在、语义作用和正常效用。
author: AI保留实例表示知识并按三层问题及实际实验反馈整理
curation_status: source_based_draft
evidence:
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 原160标签仅14个可靠断连支持；剔除重叠后164例中150例属于诱发断连
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 多架构猪基准、CClusnet、Mask2Former、BCNet、TAPPS和视频方法
    relation: references
---

## 具体缺口与当前认识

**2026-11-05更新：** 在已有V23对象记忆读出上，[功能约束后重训](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)得到形变置零+0.010234、平均共享保留+0.148434 AP（相对同预算完整重训），共享三seed同向，两种简化均过原非劣及采用护栏。这把先前低敏感性推进为“重训后仍可维持”的证据，支持降低任务保留读出的自由度，优先保留共享结构候选。当前实现还保留原卷积计算，真实结构／效率收益需随实际删除计算一起检验。

该结果限于既有内容重建与风险选择的协同，内容总量相同条件下的正确对应作用仍开放；同COCO起点的五部位辅助未稳定超过非语义分区。实例内容被使用、具名语义有特异增益和表示复杂度值得保留，现有证据对三者给出了不同判断。

中心投票、关系和对象表示都可能组织同体片段，但目标数据是否存在可靠的可见片段、普通Mask已提供多少关系，尚未确定。训练缓存断连只是标签网格性质，不能直接当物理遮挡或已观察模型错误。

[片段表示Idea](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)和[关系标注Idea](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)由此连接。[主线首试](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)已发现当前1272 TRAIN／14642原Mask仅14个可靠断连支持，低于最低128实例／32图，故结束当前缓存上的显式断片训练入口；剔除重叠诱发的150个额外断连不能替代原对象。

本次未测关系模型效用、人工成本或正常模型错误，广义归属问题仍开放。高分辨率原标注、明确可见监督或真实错误证据可以改变支持；当前没有具备条件的训练后继。认识同时回到[实例组织问题](OUTLINE_PIG_ORGANIZATION.md)。

## 1. 先定位再分割的表示

分析框与ROI如何提供对象条件，以及密集接触下的目标提议、筛选和框内干扰问题。

## 2. 从像素或局部位置形成实例

分析中心投票、嵌入、亲和关系、动态卷积和掩码组合，比较它们怎样关联同体片段、分开相邻身体，以及需要哪些监督。

## 3. 用查询或对象表示生成实例集合

分析集合匹配、对象查询、多尺度特征和Mask读出。讨论实例之间怎样区分，训练中的指派与预测反馈是否稳定。

## 4. 利用结构与关系辅助分割

比较部位、姿态、形状、边界和遮挡关系进入模型的方式。重点考察这些信息怎样影响最终实例输出，是否提供了普通实例监督没有的帮助。

## 5. 利用基础模型和额外观察

分析提示式分割、视频记忆和多视角方法。说明性能来自模型先验、提示质量、额外帧，还是场景约束。

**这里要回答：不同表示适合解决哪类错误，它们的限制来自表示、输入信息还是计算代价？**

## 项目证据如何限制结构叙事

| 对象 | 已观察到的差异 | 对后续想法的提醒 |
|---|---|---|
| V23整体—部位组合 | 正常AP正收益；固定权重形变／平均共享变化较小；新三seed重训后相对完整控制+0.010234／+0.148434 AP | 内容／外观与选择保护保留；共享保留优先进入实际结构精简，等量正确对应仍待辨别 |
| V32共享坐标对象/部位头 | 同容量竞争较独立+.2176 AP，但部位错位几乎无损，保护未过 | 不能把新颖性写成“第一次实例竞争/对象部位绑定”；新候选须超过一般owner组织 |
| 原生概念V1与新分配矩阵 | V1没有稳定C−Q增量；新矩阵直接约束原生读出但未评价 | 接口更直接是待检验设计，不能先认定它已解决知识使用 |

完整来源见[实验反馈](OUTLINE_PIG_EVIDENCE.md)。这些实验约束所属结构的解释，不要求所有未来表示都服务于合成语义。[可见片段组装](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[历史证据](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)、[可见性与歧义](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)及[按需细节](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)改变的变量不同，可以独立发展并相互启发。模型选择从所需信息和研究对象出发，不按架构名称先定答案。

## 2026-11-04内容、细尺度与选择的实际作用

[固定权重消融报告](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)在完整432dev核到：C21内容输入／细尺度残差置零分别下降2.306973／2.565348 AP；V23外观码置零下降1.315483。当前实例读出使用空间内容、细尺度重建及外观对应。细尺度移除使原N失败修复61→20、成功损伤18→0，表明表达能力与保护作用需要共同设计。

强制完整候选使两模型原N损伤升至244／178，并产生1376／1816条空Mask，提示候选内容需要通过选择和幅度控制进入输出。下一表示问题是保留内容总量后，正确空间／部位对应相对等量错配的作用；原V23保留，形变和逐部位保留进入简化候选。认识回到[实例组织问题](OUTLINE_PIG_ORGANIZATION.md)与[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，当前片段缓存入口仍按原支持不足决定暂停。

## 相关文献线索

[CClusnet-Inseg](https://arxiv.org/abs/2206.01942)、[Mask2Former](https://arxiv.org/abs/2112.01527)、[BCNet](https://arxiv.org/abs/2103.12340)、[TAPPS](https://arxiv.org/abs/2406.10114)及[猪视频基础模型系统](https://arxiv.org/abs/2604.03426)提供不同表示和信息条件的例子。分析针对具体计算和对照，不按模型名称直接排序。

[上层：实例组织问题](OUTLINE_PIG_ORGANIZATION.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [相关缺口：监督与适配的作用](OUTLINE_PIG_TRAINING.md)

## 2026-11-05反馈循环的下一判别

B组共享保留和形变置零的功能非劣继续保留；本批原卷积与输出分支仍计算，实际压缩价值需要端到端成本证据。下一步先剖析原网络、候选、采样、解码和策略的开销，再决定删除对象，核对等价性或近似损伤。实际成本回报与精度、原成功保护共同决定是否采用，本轮有限范围和实用目标统一见[报告第8.6节](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。
