---
schema_version: 1
id: OUTLINE_PIG_INSTANCE_SEGMENTATION
kind: outline
title: 真实猪场的可靠逐头实例分割
parent_id: OUTLINE_INSTANCE_SEGMENTATION
outline_level: 1
summary: 以中心命题统领五个关键研究问题，再连接具体缺口、Idea、探索主线和结果反馈；保留已排除范围与开放可能性。
author: AI依据用户2026-11-03三层整理与开展实验指令修订
curation_status: source_based_draft
evidence:
  - path: experiments/pig_multi_idea_night_20261105/CLOSURE_SUMMARY_V1.json
    revision: 1d9bc74e22082a31dc63beea890c2c86e81b50a2d46b870250c9721b48b92923
    locator: 三组Idea三seed完整训练与验收；语义增量、功能简化和局部保持的不同判断
    relation: references
  - path: experiments/pig_c21_v23_ablation_20261104/RESULT_SUMMARY.json
    revision: 652dc5f6fa03f46f73ef0b45ad735020b0ee89b1d2f87358e8024e9512e43e31
    locator: 固定权重七项功能消融及正常对照；内容、细尺度、外观与选择保护的实际作用
    relation: references
  - path: experiments/pig_appearance_stability_20261103/runs/RUN_PIG_APPEARANCE_FORMAL_20261103_b34157c23fcd/SUMMARY.json
    revision: ee0caf7ee68738376b8c94d43b78f50ccf5a5ec19b1f7b1202e42519dafd00d2
    locator: 固定几何成像首试未达预定大幅实例成功数损失；真实与合成分别解释
    relation: references
  - path: experiments/pig_fragment_relation_20261103/runs/RUN_PIG_FRAGMENT_S0_20261103_3a3e6ca63230/SUMMARY.json
    revision: 4284d2de6a6873f14503502867bb98075a099291827a40b9224c21d23c02d47d
    locator: 片段关系首试的数据支持量及预定停止出口
    relation: references
  - path: project/research_notes/sources/open_analysis_scope_4b748647b24d2dd8.txt
    revision: 4b748647b24d2dd8818d62c244551f1713dcfdd255e3c1245c351d5a3933ba57
    locator: 从问题出发，不预设模型或固定方法
    relation: references
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 已核文献、近邻做法和证据范围
    relation: references
---

## 中心命题及其与主线的关系

**怎样在真实猪场中获得准确、可靠、能够适应不同条件的逐头实例分割？哪些观察、数据、表示和学习方式值得利用？** 应用、输入和成本权衡仍可发展，不将YOLO、部位、合成或某个AP目标提升为整个命题的定义。

纲要积累认识与开放问题，从中形成持久保存的Idea；主线组织所选Idea的探索，实际结果再修正Idea及纲要。SPM／猪体知识和[片段关系主线](../../paper/manuscripts/pig_fragment_relation/README.md)都是已有探索入口，二者都不限定其他问题的外延。

成熟思想与方法资料由[知识入口](../research_notes/README.md)按问题连接，可跨纲要和Idea复用；本页保持中心问题及研究判断，完整阅读和实验依据由各自来源承载。

## 第二层：五个关键研究问题

| 关键问题 | 第三层具体缺口 | 相关Idea |
|---|---|---|
| [怎样把视觉证据组织成每一头猪？](OUTLINE_PIG_ORGANIZATION.md) | [真实错误属于哪一类](OUTLINE_PIG_TASK.md)；[同体断片怎样归组](OUTLINE_PIG_MODEL.md) | [实例信息利用](../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)、[关系标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md) |
| [判断一头猪需要哪些观察？](OUTLINE_PIG_OBSERVATION.md) | [历史可见证据](OUTLINE_PIG_HISTORY_EVIDENCE.md)；[原图细节](OUTLINE_PIG_DETAIL_EVIDENCE.md) | [历史观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)、[按需细节](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md) |
| [哪些监督和知识值得获得？](OUTLINE_PIG_SUPERVISION.md) | [少量标注的信息价值](OUTLINE_PIG_REAL_DATA.md)；[受控合成的信息与认识](OUTLINE_PIG_DATA.md)；[语义、普通关系与适配的作用](OUTLINE_PIG_TRAINING.md) | [关系标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)、[受控合成](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)、[猪体知识](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md) |
| [怎样在新条件下保持能力？](OUTLINE_PIG_GENERALIZATION.md) | [覆盖、场景依赖与适配损伤](OUTLINE_PIG_TRANSFER_GAP.md) | [覆盖迁移](../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)、[受控合成](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md) |
| [什么输出才可信且有用？](OUTLINE_PIG_TRUST.md) | [可见、推断与未知的评价](OUTLINE_PIG_EVALUATION.md) | [可见性与歧义](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)、[历史观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md) |

以上是相对于本中心命题的三层结构。具体缺口按认识展开，Idea可跨问题关联，不必为每个分面新开一条主线。原数据、模型、训练等知识保留在对应缺口中，稳定ID没有改变；[开放Idea与交叉联系](OUTLINE_PIG_HYPOTHESES.md)作为跨问题资料保留。

<a id="tested-ideas"></a>

## 已探索的Idea：排除范围与保留问题

| 已探索的具体Idea | 当前取舍与证据 | 仍开放什么 |
|---|---|---|
| [同COCO起点联合学习](../ideas/IDEA_PIG_COCO_JOINT_V1.md) | 完整三seed后C−Q平均−0.513151且异号，Q−R+2.473579高于C−R+1.960429；5→20轮均退化，关闭本版扩训。 | 学习起点单独改变不足；普通适配退化与什么额外信息超出非语义控制仍待辨别。 |
| [V23功能简化重训](../ideas/IDEA_PIG_V23_SIMPLIFIED_RETRAIN_V1.md) | 无形变／共享相对完整重训+0.010234／+0.148434 AP，简化三seed均过护栏；共享优先保留。 | 实际删除计算后的效率和稳定性；内容总量相同条件下正确对应与选择响应的分工。 |
| [C21细尺度保持](../ideas/IDEA_PIG_C21_FINE_PRESERVATION_V1.md) | 5201参数局部更新较稳定，KL平均仅+0.024948、状态变化0／−1／+1；关闭本版扩训。 | 保持与新增修复机会的关系；其他起点／信息／范围继续分别判断。 |
| [C21／V23固定权重功能消融](../ideas/IDEA_PIG_C21_V23_ABLATION_V1.md) | 完整432dev及独立验收通过；C21内容／细尺度和V23外观码移除分别下降2.306973／2.565348／1.315483 AP，强制候选显著增加损伤。保留原V23，形变／分部位保留列简化候选。 | 当前内容重建、外观对应与选择保护如何协同；保留总量后的正确空间／部位对应有何额外作用。 |
| [夜间局部细节训练V1](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md) | R/L/D/P各16500步及完整432dev评价完成；N为多阶段保留权重，四臂均低于N。P较同条件R提高2.515091、较D提高4.105891 AP，但仍低于N2.094301、损伤4.4685%超1%，关闭当前配置。 | 保持项有相对作用，不等于方法可采用；未排除新观察或其他细节利用方式。 |
| [原128TRAIN的局部重看V1资格](../ideas/IDEA_PIG_LOCAL_RELOOK_V1.md) | 1456个同条框75关联中仅15个MaskIoU0.50–0.75失败，低于预定32；独立复核一致，停止本池，方法未测。 | 不说明局部重看无效；训练集上接近触顶不能代替开发集精细错误分母。 |
| [成像敏感性V1](../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md) | 固定128 TRAIN的实例成功数最坏损失2／1460＝0.136986%，30组合成照明配对差异0／278；低于原5%／10%效应线，当前不据此训练保持方法。 | 已见TRAIN及IoU50低输出阈值触顶，不排除细边界、分数排序、强变化或独立新场景风险；广义条件可靠性仍未知。 |
| [当前160缓存的片段关系首试](../ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md) | 1272图／14642实例仅14例／14图有可靠原断连支持，低于预定128／32；停止本缓存显式断片训练入口。150例额外断连由剔除重叠诱发，未改作主对象。 | 未测模型效用、人工成本或真实视觉断片比例。明确可见监督、高分辨率原标注或真实错误证据可能提供新依据。 |
| [固定池质量头V25／V26](../ideas/IDEA_PIG_FIXED_POOL_QUALITY_V25_V26.md) | 均低于V23，原成功损伤1.740357%／1.552211%，超过原1%限；停止固定池质量头扩展。 | 不否定其他纠错信息。新观察或不同表示尚未被这两配置检验，同一oracle机会不足以重开。 |
| [合成共视原型对应RLPL V2](../ideas/IDEA_PIG_RLPL_V2.md) | D2较C2高2.124169点，却较共同曝光无辅助B2低1.740715、较保留N低2.862116点，损伤5.3151%；停止原三配置，B2不称纯真实续训。 | 哪种对应有价值、迁移与正常能力保持怎样作用，仍未唯一解释。 |
| [原生部位V1及受限适配后继](../ideas/IDEA_PIG_NATIVE_PART_V1.md) | C较N均值低0.848924点，C−Q仅0.088211且seed异号，损伤约3%；关闭V1。重放未满足V2必要条件，V2未训练。 | 不否定所有猪体知识。语义冗余、真实迁移与任务利用仍需区分，不靠继续加loss重开本版。 |

这些停止结论限定在原版本和比较条件，不是对整个中心命题的否定。[既有分配矩阵](../ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)尚无正常评价结论；2026-11-03已恢复服务器并核到CP／QP调度及2000末点元数据，原暂停期间的未知保留为历史事实，最新情况见所属记录。

## 2026-11-04综合复盘回到中心命题

[本轮总结](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)没有新增实验，但校正了认识：保留N是多阶段目标任务权重，普通续训R是另一参照。[夜间P](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)高于R2.515091 AP而低于N2.094301，保持项的相对作用与模型采用价值必须分开；[部位V1](../ideas/IDEA_PIG_NATIVE_PART_V1.md)的C−Q不稳定、[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)的D2低于共同曝光无辅助B2，继续限制各自方法主张。

这次反馈降低的是“低于N即可认定新增方法负效应”“只训练新头或BN统一解释失败”的可信度，没有为旧方案翻案。[学习／监督](OUTLINE_PIG_TRAINING.md)、[细节观察](OUTLINE_PIG_DETAIL_EVIDENCE.md)、[迁移保持](OUTLINE_PIG_TRANSFER_GAP.md)和[可信评价](OUTLINE_PIG_EVALUATION.md)分别保存这些区别；广义知识、观察和跨条件Idea仍未知，原具体版本的关闭及输入不足状态保持。

先前片段与局部首试没有方法效用结果，成像小效应也没有验证精边或新域；它们不能和三组训练合成“六个方向均无效”。需要公平普通训练对照的缺项只约束相应训练主张，不把补YOLO对照变成关系标注、历史观察或其他开放Idea的统一前置。

## 当前从Idea进入探索的位置

[夜间三组Idea完整报告](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)与独立验收已完成。它使认识进一步分化：一般辅助可改善同COCO起点的普通训练，但正确部位没有稳定超非语义控制；既有V23的任务保留图可以加强共享约束并维持精度；在C21局部范围再加保持项的边际作用很小。A、C当前版本结束，B功能简化结论保留。

当前有依据的继续问题优先是兑现共享保留／去形变的真实结构和计算节省；原V23继续作成熟参照。机制侧仍需理解相同内容总量时的正确对应与选择器响应，学习侧保留共同适配退化和语义获取／迁移／任务利用的分工。上述认识沿[学习](OUTLINE_PIG_TRAINING.md)、[表示](OUTLINE_PIG_MODEL.md)及[评价](OUTLINE_PIG_EVALUATION.md)回到监督、组织和可信性问题；历史观察、关系标注与覆盖等Idea继续按各自问题开放，实际执行安排见[总控](../ACTIVE_PLAN.md)。

以下保留前一细节、局部及成像入口的结案认识。

[按需细节新线](../../paper/manuscripts/pig_adaptive_detail/README.md)的[独立四臂训练Idea](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)现已完整结案。R/L/D/P Mask AP为52.734108／54.284877／51.143308／55.249199，均低于N57.343500；D不优于普通续训或低细节控制，P有相对保护效应但净收益和损伤护栏仍失败。四臂各16500步、全部评价／统计／终态正常完成，当前配置不继续投入；详见[结案及来源](../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)。本广义方向暂缓／未知，不把具体训练版本失败推广到所有新细节或观察。


前一[局部重看最小验证](../../experiments/pig_local_relook_20261103/REPORT_V1.md)只获得15个合格失败、不足32，按原停止出口收束，没有局部新前向。夜间训练使用随后新增的1272TRAIN训练授权，属于不同命题与预算，不修改这次资格不足事实。

以下保留成像首试时的选择及取舍，不覆盖上述当前路由。

在片段缓存入口收束后，用户要求从剩余缺口中继续选择并完成首个验证。本次选[条件可靠性](../../paper/manuscripts/pig_condition_reliability/README.md)，借现成几何不变照明对照检验[具体成像Idea](../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)，而不等待未核序列或把原缓存重新切碎。

[完整验证已完成](../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)：真实主量0.136986%，合成配对差异0，均低于原效应线。因此不启动针对这组粗粒度损失的保持训练。该判断受已见TRAIN、IoU50与低输出阈值触顶限制，下一步需要更贴近真实细粒度或独立新条件的失效证据；当前没有具备条件的方法训练后继。

前一片段首试同样已结束：14／14642个原Mask有可靠断连支持，数据入口不足。两次结果各自限定排除范围，不合并成一个科学试验，也不否定中心命题。

其他[候选及交叉联系](OUTLINE_PIG_HYPOTHESES.md)继续开放：历史可见证据与可信归属、受控合成与场景迁移、按需细节与关系询问。执行结果应回写本节和对应缺口：排除了什么、依据与范围是什么、还剩哪些值得研究的问题；没有合格后继时允许保留未知。

## 支撑知识与追溯

[文献与阅读范围](OUTLINE_PIG_LITERATURE.md)、[项目实验反馈](OUTLINE_PIG_EVIDENCE.md)、[Idea演变](../research_notes/IDEA_EVOLUTION.md)提供来源。原七个分面不再作为必须按顺序完成的章节；旧论文叙事仅作局部历史材料。正式执行、原Goal／Gate、资源和暂停／接续的事实分别沿[总控](../ACTIVE_PLAN.md)及所属Study，不由纲要结构自动改变。

## 2026-11-05反馈循环后的中心路由

本Study的具体取舍已经完成反馈闭环：A当前语义联合配置停止，B共享保留图作为实际结构精简优先候选，C当前局部保持配置停止；广义猪体知识仍为unknown。中心命题没有被缩成这三种方法，关系、观察、监督、迁移和可靠性方向继续按各自证据发展。下一循环只先做已有预测的零训练动态／实例错误诊断和B的真实成本剖析，不自动追加训练、不扩大seed、不打开test426或旧分配矩阵。新增证据或有依据的信息假说可形成有限后继，结果再回写具体方法和更大命题的判断；本轮范围与待接续状态集中见[报告第8节](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。
