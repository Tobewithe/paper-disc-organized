# 读取依赖关系的两项候选：有限文献核对

2026-10-04最新状态：候选响应共识已完成并停止配置，见[candidate_response_consensus](../../experiments/candidate_response_consensus_20261004/REPORT.md)；候选关系S/T/M已在笔记本实际运行，见[candidate_relation_readout](../../experiments/candidate_relation_readout_20261004/PROTOCOL.md)。旧S026已做GT-free邻居像素输入，本轮只是补系数链路与同容量空间对照，不能称首次。下文“未决定执行/未找到重复”保留当时推导，不能作为重新启动指示。


2026-10-03。本文下方保留当时的文献推导与候选设计，不作为待启动队列。最新去重核对：六臂快速筛选已结束；两步反馈读取和局部视图回投也已实际完成，见下列状态。当前研究授权及优先级以[目标入口](PROTOTYPE_COEFFICIENT_ACTIVE_GOAL_20261003.md)为准。

- **两步反馈已做，当前实现停止**：[实际报告](../../experiments/prototype_feedback_readout_20261003/runs/RUN_17e5267de70f4364a8fde3ecc4c7a60d/REPORT.md)与[scripts/feedback_head.py](../../experiments/prototype_feedback_readout_20261003/scripts/feedback_head.py)确认R使用更新响应、F使用初始响应并重新聚合特征，含同参数RH。dev R−F约0，R−A为−0.0575pp。不能重开同一设计，也不能由此宣布所有反馈无效。
- **局部视图已做，当前实现停止**：[实际报告](../../experiments/local_view_coefficient_replay_20261003/REPORT.md)。直接视图与回投均低于原模型，包含原图重选与概率池化/投影控制，不能再记为尚未执行。
- **候选间关系读取**：本文只提出假设；是否已有同类实际实验仍需完整去重，不能因本文没有执行链接就当作项目未做过。

2026-10-04补充去重（尚未锁定下一轮）：旧[S071源码](../../experiments/source_location_coefficient_20260913/source/source_location_coefficient_probe.py)已做同图同尺度3×3 raw系数直接平均，旧版本96目标高组IoU约−6.579pp，见[旧汇总](refinements/coco-structure/JOINT_DECODER_QUERY_RESULTS_20260913.md)。不能将邻域平均重命名成新方法。旧[候选选择设计](refinements/coco-structure/CANDIDATE_SELECTION_DESIGN_20260912.md)提过候选池/分数加权，但未找到完整运行；[官方任务对齐](../../experiments/official_coco_task_alignment_20260916/REPORT.md)中的单donor替换亦不同。当前只留下一个需要进一步文献与实现审查的问题：GT-free的实例对应/响应一致性约束，是否能使多个现成候选提供超出单donor和无条件均值的信息？尚未决定执行，不能据此自动启动平均/加权新臂。

同日[监督—评价域审计](../../experiments/supervision_output_domain_audit_20261003/REPORT.md)已完成：多数实际误删在标签一致前景，开发官方BCE也未改善；冲突区删除率较高但不支持它解释全部退化。此结果不为恢复旧选点、反馈或局部视图方案提供新的直接理由。

## 已有证据的边界

已读[原型—系数文献记录](PROTOTYPE_TO_COEFFICIENT_LITERATURE_20261003.md)、[干预去重表](READOUT_INTERVENTION_DEDUP_20261003.md)、[PCDCR](../../experiments/prototype_conditioned_dynamic_coeff_20261002/REPORT.md)、[OGPS及其补充诊断](../../experiments/ownership_solver_replay_20261002/REPORT.md)。

- PCDCR的动态低秩、原型统计条件没有建立相对静态头或错配条件的可靠优势。OGPS的GT粗网格可兑现较多oracle机会，但既有7O预测经solver仍低于原模型；不能再把“拼P/旧mask，再投影”当新问题。
- U/Q/P/L/B已完成静态读取位置比较，未显示足够新增收益。下面两项改变读取依赖关系，已有笔记和用户讨论都曾提出，不能署为本次首次发明。
- 最新joint正式结果的S−D为−0.0476个百分点、区间跨零；较早冻结模型的S>D不能升级成通用结论。以下不以“solver已被证明优越、唯一问题只剩evidence”为前提。

## 1. 历史候选：更新后的mask是否能指导下一轮读取（已有实际对照，见上方）

**原论文做了什么。** K-Net用上一轮mask对图像特征求加权聚合，再更新产生mask的kernel，并可重复这一过程；原论文还包含自适应融合和kernel间交互。Mask2Former用预测mask约束cross-attention的读取范围。这里借鉴“输出反馈到读取”这一依赖，不照搬两套完整网络或其额外模块。[K-Net §3.2，NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)、[Mask2Former，CVPR 2022](https://arxiv.org/abs/2112.01527)。

**迁移假设。** 首次读出的错误区域划分，可能使一次性的原型条件失去价值；第一轮形成的Pc1若能改善实例范围，第二轮据此读取原生H/F，可能得到更有用的32维系数。这不是“多算一遍肯定更好”。

\[
c^0\rightarrow Pc^0\rightarrow\operatorname{Read}(F)\rightarrow c^1
\rightarrow Pc^1\rightarrow\operatorname{ReRead}(F)\rightarrow c^2.
\]

**与旧实验的区别。** PCDCR/7H有原mask条件，但没有根据本次产生的c1再更新读取；当前点头一次前向中消费当前native系数，也不等于实现了这个闭环。建议保留相同P、框、原生分支与轻量系数维数，最多两轮、两轮共享同一套参数，不加更宽头或新监督。

**一小时级的判别方案。** 只比较两个同预算训练组：第二轮根据更新Pc1读取，与第二轮仍根据Pc0读取；两者都执行两轮、参数量相同、训练损失相同。读取采用软权重，保留原框全域通路，避免初始漏掉的区域被硬mask永久屏蔽。固定1024 fit/256 dev、一个seed、固定短预算；实际一小时不能完成就记为未完成，不自动延长或削样追阳性。不得用“两轮比一轮”单独证明反馈有效。

**停止。** 更新反馈未胜过同预算固定反馈，或正常掩码没有达到当前screen预设的实质效应量级，就停止这个两轮版本；不自动加第三、第四轮，不用内部loss下降代替正常掩码效用。初始误分导致自我强化是主要风险。短训阴性仅否定该有界实现，不证明所有迭代读出无效。

## 2. 次优先：让系数读出看到“其它候选如何解释同一像素”

**原论文做了什么。** K-Net在更新kernel后，用multi-head attention及FFN进行kernel间交互，使一个kernel可以获取其它group的上下文。这不是要求系数互相排斥，也不是强制实例mask互斥。[K-Net §3.2，Kernel Interaction](https://proceedings.neurips.cc/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)。

**迁移假设。** 当前独立读取能知道“我的框里有什么”，却没有显式看到“附近另一候选已如何解释这片区域”。可将自身响应与少量邻近候选的prototype响应一起送入同一个空间证据读出，再产生当前c；不直接对不同c施加排斥约束。

最小实现可使用每个读取位置的竞争响应/相对响应，而不是整套更大的跨候选Transformer。例如额外输入邻居响应的汇总及自身—邻居差值；真正要测试的是新增关系信息，非增加头宽度。所有关系组保留相同输入通道、参数与预算。共享P自然把不同候选的响应放在同一像素坐标中，避免直接比较c向量距离。

**输入边界。** 邻居必须来自原模型冻结的、无GT的预测池，例如原有head输出保留规则下的候选；不得把TAL正样本列表当推理邻居。预测重复可能是同一实例，合法mask也可重叠；不做像素级跨候选softmax排他分配。普通实例BCE本来已经将当前GT之外像素作为负例，不能以“原监督没有实例身份”作为动机。

**一小时级的判别方案。** 同容量的自身输入控制、真实邻居、关系错配三个短训组；错配保持自身query/P/框不变，并保留邻居数量与可用性记录。已有缓存不足以提供无GT邻居时，先生成固定池，不能偷换为TAL列表。只在固定1024/256、固定短预算内筛查，不扫邻居数或关系层数。成本若超过一小时，不挤入本次预算。

**停止。** 真实关系不优于同容量自身控制及错配，或仅在极少数对象有变化且达不到所声明任务范围的实质收益，就停止；不追加对比loss、gate或更多关系层。true>wrong单独只能证明输入依赖，仍需超过自身控制与原模型。关系错配也会改变分布，不能独立作为充分的因果证据。

## 本次没有提升优先级的第三来源

[The Fast Bilateral Solver，ECCV 2016](https://arxiv.org/abs/1511.03296)用图像颜色/位置亲和性构造边缘感知平滑与数据保真最小二乘。它可以启发在P空间中加入像素关系约束，但当前没有证据说明简单平滑能修复主要失败；纹理边缘、遮挡、同色邻居还可能误导它。本次不因此追加图正则、RGB输入或求解器参数扫描。

这段历史构思关注两个依赖关系：**本实例前一轮输出→下一轮读取**，以及**其它候选解释→当前实例读取**。二者均有成熟文献先例；前者的具体两步对照现已完成，后者执行前仍需历史去重。原创性、适用失败类及有效性不能由此自动成立。

## 补充候选：预测框局部视图的空间证据，回投原图原型

后续窄查未找到完整已执行对照。已有[局部重推](../../experiments/failure_object_roi_probe_20260914/REPORT.md)用GT框定义ROI；[小目标输入干预](../../experiments/small_object_causal_interventions_20260914/REPORT.md)主要解释P3框定位；OGPS使用GT/7O图而非新图像视图。因此只能说此组合尚未找到重复，不能称首次裁剪或首次投影。

待检验链路是：原预测框→冻结官方模型的局部放大视图→无GT候选对应与几何还原→以该连续响应为证据求原图固定P中的系数。框和原图P保持原样，GT仅评价，不借GT框或GT类别选局部预测。它改变证据来源，而不是继续加宽同一ROI读出；会增加推理成本，必须报告。

最小区分应保留原输出、自回投零变化检查、局部视图直接mask、局部视图投回原P四种结果。若局部视图本身没有有效信息，不训练新的适配器补救；若直接视图有效而投影损失收益，则问题在信息压缩/求解与支持；若两者均有效，再判断为何值得保留32维系数接口，而不能把TTA收益冒充投影独立贡献。局部候选身份可能错配、上下文可能丢失、分数尺度也会改变；这些是尚需验证的风险，不能预判该方案会成功。

[官方CVPR2020 CascadePSP论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_CascadePSP_Toward_Class-Agnostic_and_Very_High-Resolution_Segmentation_via_Global_and_CVPR_2020_paper.html)提供全局/局部高分辨率分割细化的成熟先例；该文使用额外训练的细化模型，没有证明上述冻结YOLO+原图P回投链路。这里只借鉴局部图像可提供额外空间证据的研究思路，不引入其网络、不作新颖性断言。本项后来已登记并完成，实际实现、对照及阴性结果以上方local-view报告为准。
