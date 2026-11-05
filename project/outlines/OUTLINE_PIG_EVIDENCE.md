---
schema_version: 1
id: OUTLINE_PIG_EVIDENCE
kind: outline
title: 资料：猪实例分割既有探索与实验反馈
parent_id: OUTLINE_INSTANCE_SEGMENTATION
summary: 保留已有实验事实及适用范围，让内容依赖、监督语义、可控能力与迁移问题启发不同Idea，不把局部主线等同于中心命题。
author: AI依据本地原记录及2026-11-03已回传核验的服务器产物整理
curation_status: source_based_draft
evidence:
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 已完成成像条件首试，原效应线未支持；不改变正常任务评价或旧Gate
    relation: references
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 当前TRAIN原断片支持不足；未测方法效用
    relation: references
  - path: experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json
    revision: 0b04a8d7d5006c2e030c5afeb8f719118b1c2730328c7297f2daabfc5d46576b
    locator: CP/QP各2000末点元数据及两作业accounting只读对账
    relation: references
  - path: project/research_notes/sources/concept_native_v1_report_71fdc28133cda7a2.txt
    revision: 71fdc28133cda7a23fe99155240aea6d7343e9d5d3d059d4e06e5e0a9c6d4039
    locator: 四臂三seed正常评价与有限重放
    relation: references
  - path: project/research_notes/sources/assignment_bridge_pause_7e86dc12ee5f9b0e.txt
    revision: 7e86dc12ee5f9b0e34995982c582e925bed004c8880982c4824879780ab065c7
    locator: 暂停记录、四份末点元数据及两作业未核终态
    relation: references
  - path: project/research_notes/sources/concept_synthesis_release_72908863d895ceae.txt
    revision: 72908863d895ceaea2354b15632013bebe473d7f3e1124a634e559190ea2a94f
    locator: 有限数据发布与标签资格
    relation: references
---

## 与问题纲要的关系

本页保留项目探索和真实进度，供[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)按适用范围参考。原报告和Run保存证据，经过判断的停止范围、反例与开放问题直接[回写中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，让纲要继续产生Idea，再由主线探索并反馈。其执行状态和既有方法不决定纲要应选择何种模型、监督或研究方向。

已补建[固定池质量学习](../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md)、[RLPL V2](../ideas/IDEA_PIG_RLPL_V2.md)与[原生部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)三个具体Idea的取舍入口，均明确为据原报告事后整理，未改原科学结论、协议或Run。

## 已有探索分别说明什么

| 探索与原入口 | 实际状态与认识 | 结论边界 |
|---|---|---|
| [同COCO起点联合学习](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md) | 三臂各3seed／2120步完成；C−Q平均−0.513151且异号，C−R+1.960429而Q−R+2.473579；5→20轮三组均退化 | 关闭本版语义联合扩训；保留一般辅助与共同适配退化认识 |
| [V23功能约束重训](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md) | 三臂各3seed候选／成本／风险完整；无形变／共享保留相对完整重训+0.010234／+0.148434，共享三次正向、简化全部过护栏 | 功能简化获支持，优先共享；当前还未删除实际计算，效率仍待兑现 |
| [C21细尺度保持](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md) | 两臂各3seed／4000步；KEEP−FINE+0.024948且异号，GT75变化0／−1／+1，均较原C21略低 | 局部稳定保留，保持增量不足，关闭本版扩训 |
| [C21／V23固定权重功能消融](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md) | 完整432dev十组输出、七项干预与1176独立验收完成；C21内容／细尺度−2.306973／−2.565348，V23外观码−1.315483，强制候选两组−4.777370／−3.750892 AP | 当前固定模型依赖内容重建、外观匹配与选择保护；形变／共享保留图变化较小，进入简化候选 |
| [局部重看V1病例资格](../../experiments/pig_local_relook_20261103/REPORT_V1.md) | 128TRAIN／1460GT、1456同条框75关联，合格失败仅15例，不足32；独立复核误差0，6CPU-core秒、0新前向 | 当前池支持不足、方法未测；不改分母，不外推为所有局部观察无效 |
| [夜间四臂真实细节训练](../../experiments/pig_detail_training_night_20261103/REPORT_V1.md) | 1165–1171全部完成，四臂各16500步；R/L/D/P Mask AP52.734108／54.284877／51.143308／55.249199，均低于N，当前联合继续线全部失败 | P较D有+4.105891相对效应，但P−N为−2.094301、损伤4.4685%仍超限；保留局部作用，关闭当前配置，不外推否定所有细节 |
| [固定几何的成像敏感性V1](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md) | 工程1158与正式1159完成，128真实TRAIN最坏实例数损失2／1460，30组合成照明配对差异0／278；不据此训练保持方法 | 已见TRAIN、IoU50和低输出阈值触顶；不是AP或独立泛化，合成只在辅助有效域诊断 |
| [TRAIN片段关系适用性](../../experiments/pig_fragment_relation_20261103/REPORT_S0.md) | 全1272图／14642实例，仅14例／14图可靠原断连支持；低于128／32最低量，停止当前缓存入口 | 数据适用性限制，不是模型、人工成本或真实可见断片比例的效用试验；150额外断连由剔除重叠诱发 |
| [合成概念标签修复](../../experiments/spm_concept_synthesis_repair_20261101/REPORT.md) | 90图发布完成，72训练/18QA，几何与标签资格有据可查 | 不代表完整猪场分布、真实概念准确率或AP收益 |
| [V21/V23正常输出](../../experiments/spm_content_constrained_memory_20261026/REPORT.md) | V23相对N+3.409778 AP、损伤18/2126；正常收益保留，内容／选择的最新分解见本表本轮消融 | 原收益及旧错配结果按原控制保存，新功能依赖证据另由本轮Study承载 |
| [固定候选机会](../../experiments/spm_content_opportunity_20261026/REPORT.md) | GT有限见证N+6.118060、零损伤，闭合该有限池上界 | 不是部署分数，也不是当前原生概念模型的表达上界 |
| [V25动作质量](../../experiments/spm_action_quality_20261026/REPORT.md)及[V26局部特征质量](../../experiments/spm_native_spatial_quality_20261026/REPORT.md) | 均未超V23，损伤超限，固定候选质量头扩展已停止 | 不能仅凭上一行oracle机会大便再开更大选择器 |
| [V32共享坐标竞争](../../experiments/spm_registered_object_head_20261027/PROTOCOL_V1.md)及[原评价](../../experiments/spm_registered_object_head_20261027/delivery/evaluate/runs/RUN_SPM_REGISTERED_EVALUATE_4faf44ba1797/SUMMARY.json) | 竞争较同容量独立+.217605 AP，正确−部位错位−.001038；损伤58/2126，整体不采用 | 普通竞争相对作用、有序部位必要性与正常可采用性是三件事 |
| [RLPL完整流程](../../experiments/spm_rlpl_pipeline_20261101/FINAL_REPORT.md) | D2−C2+2.124169 AP，但D2−N−2.862116、损伤5.3151%，冻结筛选阴性 | 不否定所有对应/原型，也不将对较差控制的优势当净效用 |
| [原生部位V1](../../experiments/spm_concept_native_training_20261102/FORMAL_REPORT_V1.md) | 四臂三seed及重放完成；C未稳定超过Q，保护失败；同原检测重放也未给可用增量 | 只约束原数据、接口和预算；合成QA可读不等于真实概念已充分学会 |
| [分配—部位—Mask有限判别](../../experiments/spm_concept_assignment_bridge_20261102/PROTOCOL_V1.md) | 六臂各2000末点元数据均已核，2026-11-03补核CP/QP；评价仍暂停 | 暂停不是阴性，训练完成也不是效用成立 |
| [分配实际变化的日志核读](../research_notes/PIG_NEXT_RESEARCH_PRIORITIES_20261102.md) | CM/QM真实O2O累计raw状态变化仅4/2；Q loss下降，指定层五时点Q/C梯度比中位约1.014 | 只读TRAIN元数据；不是唯一猪数或独立比例，不能事后改Gate，也未裁决六臂科学效用 |

V1中正确部位C平均Mask AP56.494576，原生N57.343500；C−Q仅+0.088211且seed异号。该结果与当前暂停矩阵不能合并算成一次试验，不能跨版本拼接成功。旧报告中的历史建议也不自动决定最新行动。

## 2026-11-04综合认识已回到Idea与问题纲要

[综合复盘](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)没有新增实验；它将基线身份和比较权限的纠正写回[夜间](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)、[部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)、[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)及三个首试Idea。N是多阶段保留权重、P−R为正而P−N为负、B2有合成曝光等事实，分别进入[训练](OUTLINE_PIG_TRAINING.md)、[细节](OUTLINE_PIG_DETAIL_EVIDENCE.md)、[迁移](OUTLINE_PIG_TRANSFER_GAP.md)和[评价](OUTLINE_PIG_EVALUATION.md)，并回到[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)。

这次改变的是解释，原Run、数值Gate及具体Idea状态没有变化；BN／训练整合尚不能作为统一根因。六项原结果不是六次新的证据，也不把方法未测并作训练阴性；主线入口和广义Idea保留同一范围，工作台是否同步另行说明。

## 2026-11-04本轮功能消融改变了哪些认识

[完整报告](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)与[验收回执](../../experiments/pig_c21_v23_ablation_20261104/ACCEPTANCE_RECEIPT.json)将旧的“内容破坏控制较弱／不足”推进为更具体的判断：重建器内容输入、细尺度修正和外观码在当前正常路径有实际作用；强制完整候选显著增加损伤并产生空Mask，选择保护成为解释已有收益的关键部分。形变及逐部位保留的变化较小，研究优先级降低。

新结果进入[具体消融Idea](../ideas/IDEA_PIG_C21_V23_ABLATION_V1.md)及[广义猪知识](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)，沿表示、学习和评价缺口反馈到相应关键问题及[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。下一科学问题聚焦等量内容下的正确对应与选择器响应，原V23保留，既有失败版本及暂停矩阵保持原安排。

## 2026-11-05实际训练与交付验收

[新夜间Study](../../experiments/pig_multi_idea_night_20261105/study.json)在原固定模型敏感性基础上检验了不同学习起点、重训补偿和局部保持。原79任务、新1258／1259验收全部完成，24配置共102192正式更新；完整432dev、45终点／参照的1080数值及128835状态独立重算误差0，24最终包各两RGB的978输出精确重放。实现、来源、指标及交付通过，与各Idea的研究判断分别保存。

新认识进入[学习](OUTLINE_PIG_TRAINING.md)、[表示](OUTLINE_PIG_MODEL.md)、[评价](OUTLINE_PIG_EVALUATION.md)与[适配](OUTLINE_PIG_TRANSFER_GAP.md)，沿监督／组织／可信性与中心纲要积累。当前优先保留共享功能约束用于后续实际压缩，A／C本版扩训结束；原V23、内容正确对应问题及其他开放Idea保留。

## 从既有结果打开新的问题

可靠标签、概念可读、正确内容用于实例区分与正常任务效用需要分别认识。这限制原概念路线的解释，不能升格为整个中心命题只能研究的一条链。

已有事实可以导向不同想法：内容归因不足提示比较普通关系与具名语义；标签语义差异提示[可见证据与歧义](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)；可控release提示[合成认识工具](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)；有限真实迁移提示[跨条件可靠性](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)。[关系标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)、[片段归组](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)和[额外观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)也可借文献与这些教训形成独立思路，尚未被表内实验验证。

上一轮聚焦论文的组织与三候选排序保留为历史理解，已由用户明确纠正。新的[开放探索图](OUTLINE_PIG_HYPOTHESES.md)不预先将所有方向压成一个方法，也不把原矩阵未决设为继续构思的障碍。2026-11-03已有两个有限入口形成真实结果：片段缓存支持不足，以及本范围成像条件的大幅粗粒度损失未获支持；相关广义Idea仍各自未知。

## 原分配矩阵的暂停与最新对账

2026-11-02[暂停记录](../../experiments/spm_concept_assignment_bridge_20261102/PAUSE_V1.json)保留当时四臂已核、两作业未核的事实。2026-11-03依据新用户任务开展了只读对账，见[RECONCILIATION_20261103.json](../../experiments/spm_concept_assignment_bridge_20261102/RECONCILIATION_20261103.json)：CP/QP的1154/1156均COMPLETED／0:0，原Run与SUMMARY确认各2000末点，四文件原SHA核验回传。

六臂训练已核，权重仍留服务器，dev未评价、后两seed未准入、效用／机制未知，test426封存。本次执行的是独立片段关系首试，没有恢复旧矩阵评价或重复提交训练。对应[Idea](../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)继续为已暂停／判断未知，原Run、Gate和预算不追改。

## 反馈怎样进入知识

实验结束后，应把实际收益、损伤、反例及限制反馈到对应问题纲要；不能由退出码或Run完成自动把Idea改成支持。新数据资格、知识学会、任务使用与正常效用也分别判断。

工作台中的[本次结构与证据反馈记录](../research_notes/PIG_KNOWLEDGE_STRUCTURE_20261102.md)提供显式关联，便于从纲要和Idea回到原证据。最新执行与后续恢复条件仍以[研究总控](../ACTIVE_PLAN.md)和所属Study为准。
