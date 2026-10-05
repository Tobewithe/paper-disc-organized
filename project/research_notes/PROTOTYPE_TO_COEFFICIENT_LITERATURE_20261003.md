# 原型参与系数生成：文献启发与历史去重

2026-10-03。响应用户“按照原型到系数产生的链路，从以往文献寻找新的想法”。本记录是研究讨论，不是训练协议；没有新建实验或启动计算，不改变正在执行的原生系数分支联合训练。

## 研究对象

用像素为行的约定，P∈R^(HW×32)，c_i∈R^32，z_i=Pc_i。原生结构中，原型与系数由各自分支产生，最后组合；这里拟议的是让当前图像的原型响应更直接参与实例系数的形成。

当前空间投影方法已经实现 F/box/query/current-logit → spatial residual → prototype operator K → coefficient。新的机制不能只是再拼接P统计、扩大ROI或把同一个最小二乘写成“多区域共识”。

已有oracle支持特定候选上的可恢复机会；不等于原型对所有失败都充分。冻结版本中S相对D的小幅优势、正确框相对错框优势，均不能证明求解器或空间证据的唯一责任。正在进行的联合训练还要区分冻结原生系数分支带来的限制。

## 历史去重

| 内容 | 已有记录 | 本次不能重复声称 |
|---|---|---|
| 原mask作为输入 | [7H](../../experiments/coefficient_raw_mask_condition_20260925/RESULTS.md) | 第一次加入z0条件 |
| mask加权原型统计、动态低秩系数 | [PCDCR](../../experiments/prototype_conditioned_dynamic_coeff_20261002/REPORT.md) | 第一次使系数依赖P |
| 系数分支完整空间图ROI | [7J](../../experiments/coefficient_spatial_head_information_20260926/RESULTS.md)及后续BGCR | 第一次利用预测框区域信息 |
| 空间证据经P投影成系数 | [9/23投影](../../experiments/response_projected_coeff_20260923/REPORT.md)、[OGPS](../../experiments/ownership_solver_replay_20261002/REPORT.md)、[10/03冻结训练](../../experiments/box_evidence_projection_20261003/REPORT.md) | 第一次以空间约束求系数 |
| 少量GT像素提示 | [S046](refinements/coco-structure/SPARSE_IDENTITY_RESULTS_20260912.md) | 第一次用少数像素约束32维系数 |
| 重要格点误差传播 | [OGPS误差传播](../../experiments/ownership_solver_replay_20261002/runs/RUN_ec575d4cc7a341a0bb98b8b5cff1cf63/ERROR_PROPAGATION_REPORT.md) | 已证明GT-free选点有效；旧公平能量匹配未达门槛 |
| 2×2区域多套系数 | [空间读出对照](../../experiments/coco_spatial_readout_controls_20260922/REPORT.md) | 分区产生系数就是新机制 |

用户此前28b6c925附件已经提出反馈迭代、稀疏证据和多区域共识。本次只深化其中的可检验区别，不将它们重新署为首次提出。

## 方向一：让原型选择值得读取的空间证据

**文献来源。** PointRend（CVPR 2020）在不确定位置自适应进行点预测；AdaMixer（CVPR 2022）让query决定空间与尺度采样。Joshi与Boyd的Sensor Selection（IEEE TSP 2009）研究有限线性测量如何更有效地约束未知参数。三者都没有直接给出本项目的YOLO原型选点方法，下面是组合这些思路后的研究假设。

- [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf)
- [AdaMixer](https://arxiv.org/abs/2203.16507)
- [Sensor Selection via Convex Optimization](https://web.stanford.edu/~boyd/papers/sensor_selection.html)

**新的具体区别。** 当前固定16×16网格主要按几何均匀读取；P通过原响应和最后的K参与，但没有依据P的32维响应方向来选择读取位置。对某个像素u，z_u=p_u^T c。若许多位置的p_u近似相同，它们对系数给出的线性约束重复；多读取这些位置未必消除剩余系数歧义。

建议链路：

> 预测框内的P与当前响应 → 选择有互补约束的少量位置 → 读取这些位置的视觉特征并预测局部修正证据 → 用实际P解出32维系数 → 正常解码。

选点同时需要：实例相关性、观测可靠性、原型响应互补性。只找高不确定性可能反复选到同一类方向，只找高杠杆点可能选中视觉证据最不可靠的位置。

一种几何设计参照是在固定正则R下，增加使下式增长较大的测量：

\[
\log\det\left(\eta R+\sum_{u\in\Omega_i}w_u p_up_u^T\right).
\]

它不是已锁定的损失或采样规则。传感器选择的噪声假设不能直接搬作神经网络的掩码收益保证；若更换原型坐标，正则与归一化也须一致处理。32个未知数不意味着任取32点就能稳定或正确求解。

**信息从哪里来。** P负责指出哪些约束互补，真正的目标/修正值由无GT视觉特征预测。GT只在训练监督与评价使用。不能把GT提示实验的收益当作这一预测器已经可实现的效果。

**最小有判别力的比较。** 相同读取点数、特征来源、证据预测器、求解器与训练预算，比较均匀选点、不确定性选点、加入P响应互补性的选点。若同时新增更高分辨率特征，各组都应得到同一来源，避免把新信息源收益归给选点机制。

**停止含义。** 只有正常原图掩码与修复/损伤优于同预算对照，才支持该读出机制。只改善矩阵条件数、采样点损失或GT辅助上限不算成功；若不超过不确定性选点，不继续追加采样器或损失。

## 方向二：用更新后的mask重新读取，再产生系数

**文献来源。** K-Net（NeurIPS 2021）的kernel update使用当前mask聚合特征并迭代更新；Mask2Former（CVPR 2022）使用预测mask约束注意力读取。这支持“分割结果参与下一轮信息读取”的设计，但不是直接照搬其整网。

- [K-Net](https://proceedings.neurips.cc/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)
- [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)

\[
c_i^0\to Pc_i^0\to\operatorname{Read}(F)\to c_i^1
\to Pc_i^1\to\operatorname{ReRead}(F)\to c_i^2.
\]

真正区别是第二轮依据z1重新读取。7H、PCDCR和当前联合训练都有原始/当前一次前向的mask条件，但没有这条更新后再读的闭环。用户此前已提出此方向，本次只是补充具体文献和判别控制。

最小控制：两轮同参数规模、同计算预算，一组第二轮使用更新z1，另一组重复使用旧z0。否则“两轮优于一轮”不能区分反馈与单纯增加计算。初始漏检区域仍应保留读取通路，不能用硬mask彻底锁死框内未预测区域。

MP-Former（CVPR 2023）提出训练期带噪GT mask的辅助引导路径，也提示早期mask反馈可能不稳定。但它不是无代价修补，不据此自动增加教师损失：
[MP-Former](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MP-Former_Mask-Piloted_Transformer_for_Image_Segmentation_CVPR_2023_paper.pdf)。

## 方向三：让相关候选联合决定系数，而非独立读出

**文献来源。** Slot Attention（NeurIPS 2020）在输入与多个slot之间形成竞争性分配并迭代更新；K-Net还包含kernel之间的信息交互。可借鉴的是跨实例读取时的信息交换，而不是直接拉开32维系数。

- [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
- K-Net来源见上。

拟议对象为：

\[
c_i'=f\big(h_i,B_i,Pc_i,\{B_j,Pc_j,h_j\}_{j\in\mathcal N_i},F\big).
\]

同一像素受到多个候选响应时，当前候选可以读取对方的响应与query，形成自身的空间证据，再产生c。与专家所提“再加实例对比损失”的区别在于：改变推理时可用的信息及跨候选交互。原实例BCE本来就以当前实例GT为正、支持区内非自身像素为负，并不缺少实例标签。

**重要边界。** 邻居输入必须来自GT-free模型预测池，不能用官方TAL正样本列表当作推理可用邻居，否则GT影响了哪些实例进入输入。重复预测不等于不同GT；实例mask允许部分合法重叠，不能强制每个像素在所有候选间互斥，也不能把c向量互斥当作mask身份分离。

比较真实关系与相同模块独立读取/适当错配关系，并检查收益是否集中在确有邻居混淆的对象。若只在非常少的样本成立，只报告其适用范围；不能把全体失败改写成密集场景问题。

## 当前判断与不追加的内容

从本项目已做内容出发，**方向一的新增问题最明确：P是否应在视觉证据读取之前决定测量位置，而不只在读取之后完成投影。** 方向二文献依据更成熟、实现改动较小，但本身已有成熟先例，且用户已经提出。方向三改变了独立实例读出的假设，但风险和输入泄漏控制更多，应以邻居混淆证据为前提。

原型通道共享/置换等变网络尚未找到已执行实现，但不作为首选：原网络通道具有共同训练的身份，代数上可换基不证明实际跨图通道漂移，也不证明强制等变会改善当前泛化。

这些是有文献依据、经历史去重的候选机制，不是已确认的新颖性或有效性结论。当前先完成已有联合训练；本记录不授权并行增加三条训练线，不修改其预注册比较和停止条件。

## 后续用户授权（2026-10-03）

用户明确要求：“等目前这个实验做完后就按照你说的进行”。据此，当前联合训练与既定评价完成后，先汇总其结果，再执行优先方向一“原型引导的证据选取”。本授权取代上文仅讨论、尚未执行授权的状态，不改变正在运行实验的方案或快照。

只开展一轮有边界的比较：相同读取点数、特征来源、证据网络、求解器和训练预算，比较均匀选点、不确定性选点、加入原型响应互补性的选点。继续复用经审计的数据与官方候选身份，正式执行前锁定实现、预算、指标与结束条件。按正常原图掩码及修复/损伤判断，不用数值条件改善代替任务收益；不自动扩展反馈迭代、跨实例交互、门控或新损失。

运行仍在远端GPU或笔记本ssh 28358lan；桌面只做代码、记录和传输，SSH子进程保持隐藏。该轮完成并报告后停止，不因阴性结果自动追加补丁。
