---
schema_version: 1
id: OUTLINE_PIG_EVALUATION
kind: outline
title: 可见证据、推断与未知怎样对应可信评价？
parent_id: OUTLINE_PIG_TRUST
outline_level: 3
summary: 随研究问题选择有效证据，区分分割效用、信息价值、人工成本、可靠性与泛化；已有实验口径不自动约束所有新Idea。
author: AI保留评价知识并按三层问题及具体主张整理
curation_status: source_based_draft
evidence:
  - path: project/research_notes/sources/pig_literature_evidence_137c841822ba189a.txt
    revision: 137c841822ba189a42ffb2e2b5ed1efc06544c99a98bc5f51f9ca4d7f012457e
    locator: 原文中的指标、输入条件、数据划分和归因边界
    relation: references
---

## 具体缺口与当前认识

**2026-11-05更新：** [三组Idea的三seed验收](../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)区分了整体辅助作用、语义增量、功能简化与模型采用：C−R为正而C−Q为负，B共享保留相对完整重训三次正向但比成熟V23平均仅高0.063368 AP，C保持项只有0.024948。完整B控制首seed还出现226条空Mask并损伤22／2126原成功，略超1%；两种简化三seed均过护栏、空Mask为0。

本批选择同时依据固定终点、配对seed、原成功状态和真实成本。功能约束的非劣成立，实际删去计算后能否节省算力仍是下一验证对象；同源四簇的描述性重抽也与独立场景确认分开。相关取舍反馈[可信输出](OUTLINE_PIG_TRUST.md)及中心纲要。

真实独立标注、合成可见标签和完整身体推断的语义可能不同；需要的可信度又随计数、体尺和行为任务变化。现有有限线索不能给全数据定性，也不能通过改旧GT或评价目标掩盖差异。

[可见性与歧义Idea](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)联系[历史观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)与[按需细节](../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)。输出效用、机制和证据权限在各自范围评价，并回写[可信性问题](OUTLINE_PIG_TRUST.md)。以下保留公平比较与证据质量的支撑知识。

## 1. 指标是否对应研究主张？

区分前景质量、实例区分、局部边界、视频身份和下游应用效果。分析总体指标是否掩盖关键错误，针对性评价是否有合理分母。

## 2. 比较是否公平？

核对数据、标签、预训练、输入分辨率、参数、训练预算和选点方式。特别说明提示、已知猪数、未来帧和人工修正等额外信息。

## 3. 泛化是否得到验证？

比较同场景测试、多seed、跨视频、跨相机和跨农场证据，分析不确定性及结论适用范围。

## 4. 专用方法是否具有实际价值？

综合精度、损伤、延迟、内存、标注和数据制作成本。判断复杂方案是否优于合适的简单参照。

**本主题要理解：证据实际支持了什么，是否足以改变相应的研究判断？**

## 不同想法需要不同层次的证据

| 主张 | 必须由什么检验 | 现有反例/缺口 |
|---|---|---|
| 改善既有正常分割 | 完整正常Mask AP、原生和预定强参照、必要定位/损伤/尺度/成本 | RLPL/V32对较差控制变好仍不能采用；V23是历史强参照，异预算不作单项机制归因 |
| 正确结构有独立价值 | C/Q匹配控制、一般owner/点监督控制及真实任务差值 | V1 C−Q不稳定；V32部位错位几乎无损；不能用部位图好看替代 |
| 新因素被充分检验 | 实际干预、有效输入/梯度/覆盖及其作用对象 | M真实O2O变化很少，限制阴性解释；不能事后加旧Gate或调强追阳性 |
| 有改善空间 | 同一有限池内GT诊断并交代权限 | V23池见证+6.12不等于正常方法收益，不能移作新原生模型上界 |
| 可外推到真实新来源 | 锁定方法后按协议独立确认，训练seed和来源差异分别处理 | 432dev反复使用且仅4来源；更多seed或bootstrap不使其变成盲测 |

评价对象随问题改变。[关系标注](../ideas/IDEA_PIG_RELATION_ANNOTATION.md)还涉及真实人工时间与信息价值；[受控合成](../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)首先需要物理与标签一致、发现可重复且真实适用范围明确；[额外观察](../ideas/IDEA_PIG_OBSERVATION_INFORMATION.md)需要说明提供了什么新证据；[歧义表达](../ideas/IDEA_PIG_VISIBILITY_AND_AMBIGUITY.md)涉及可信度与具体用途。不能要求所有Idea先经过同一C/Q、分配或AP矩阵。

构思期可以保留尚无充分证据的可能性，明确未知；正式主张或执行时再匹配必要证据。原六臂读出与错误／概念小面板仅是既有主线的缺口，不是中心纲要全部探索的前置任务。旧数值Gate和test426隔离仍在所属范围保持。

## 2026-11-04由实际复盘修正的比较权限

[模型履历](../research_notes/PIG_YOLO_BASELINE_LINEAGE_20261104.md)确认N为多阶段训练与分支组合后的保留权重，R为各实验内部普通续训；RLPL B2另有共同合成RGB／BN曝光。P−R为正而P−N为负可以同时成立，分别回答训练策略相对作用与能否替换模型，不应合并为“方法不如官方初始YOLO”。

[综合总结](../research_notes/PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)及对应[夜间](../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)、[概念](../ideas/IDEA_PIG_NATIVE_PART_V1.md)、[RLPL](../ideas/IDEA_PIG_RLPL_V2.md)记录已保留各自解释范围。片段和局部资格不足仍是方法未测，成像小效应仍限既定读数；本次文字纠正不是新增独立实验，原Gate、分母与阴性不改，认识回到[可信输出问题](OUTLINE_PIG_TRUST.md)。

## 2026-11-04功能依赖与修复／损伤的联动

[C21／V23本轮评价](../../experiments/pig_c21_v23_ablation_20261104/REPORT_V1.md)固定checkpoint、完整432dev、十组各12899条输出，逐条保持检测身份，原三模型RLE精确重放。独立验收重算162个数值及28630个GT75状态，误差0；七项干预均实际改变输出。评价范围直接对应当前固定模型的功能依赖。

C21 top1全强度／V23完整候选直接应用时，修复原N失败由61／64增至87／85，成功损伤由18／18增至244／178，且出现1376／1816条空Mask。修复数与损伤一起解释了选择保护的价值。共享保留图AP提高0.075585，同时AP75下降0.060040、原V23成功少1个，方法取舍应同时查看平均精度和实例变化。

四来源重抽的描述性百分位由原匹配重新插值复核，保留307／31／48／46图的来源组成。当前保留V23，形变和共享保留图进入简化候选；上述评价认识回到[可信输出问题](OUTLINE_PIG_TRUST.md)及[中心纲要](OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。

## 分析时的口径

论文自报数字应在各自任务和信息条件内解释。不同论文的AP、F1、PQ、mIoU与J&F不直接排榜；带GT提示或已知数量的任务也不冒充未知数量自动分割。需要对比时核对共同评价对象，再解释差值。

具体项目实验仍以各自冻结协议为准，综述不追改原Gate。相关文献比较条件见[原文证据卡](../research_notes/literature/pig-instance-open-analysis_2026-11-02/EVIDENCE_CARDS.md)。

[上层：可信输出问题](OUTLINE_PIG_TRUST.md) · [中心命题](OUTLINE_PIG_INSTANCE_SEGMENTATION.md) · [资料：跨问题Idea联系](OUTLINE_PIG_HYPOTHESES.md)

## 2026-11-05结案门槛补充

本批已完成固定终点、配对seed、原成功／空Mask状态、独立数值／GT75核验和最终包RGB重放；因此科学事实与方法取舍可结案。仍需分开保存的证据边界是：432dev被重复使用且只有四个来源簇，来源重抽为描述性敏感性，不替代独立新场景；B功能非劣也不替代实际算力测量。下一轮评价把AP、损伤、空Mask、实例错误类型、时延和显存放在同一判断中，任一关键护栏失败即停止对应方法。
