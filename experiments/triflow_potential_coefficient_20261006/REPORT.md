# TriFlow 首次可行性报告

本次执行有效，未达到预先锁定的可行性门槛。

生成时间（UTC）：2026-10-07T06:21:17.290929+00:00。执行有效性：**通过**；科学门槛：**未通过**。

| 完整original COCO5000 Mask指标 | 冻结官方 | 冻结官方 + TriFlow | 增量（AP点） |
|---|---:|---:|---:|
| AP | 43.6854 | 43.7418 | +0.0565 |
| AP50 | 66.3537 | 66.3960 | +0.0423 |
| AP75 | 47.2948 | 47.4597 | +0.1649 |
| APsmall | 22.4018 | 22.3135 | -0.0884 |
| APmedium | 47.6702 | 47.7814 | +0.1112 |
| APlarge | 63.5717 | 63.6979 | +0.1262 |

以上AP展示乘100；gate使用原始AP fraction，0.003等于0.3个AP点。未知不填写0。数值oracle/20项数学诊断通过不能证明AP改善。

## 实际方法与预算

冻结官方YOLO26m-seg（Ultralytics 8.4.100），官方权重SHA256：`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。所有官方参数与BN保持冻结；仅训练新TriFlow模块。输入是真实P3特征、32通道prototype、原生one2one系数、最后系数卷积的真实hidden query及预测邻居。未使用ACD微调checkpoint。

1. Conflict Tokenizer：64个预测位置，默认32边界/16实例竞争/16prototype leverage；无预测邻居时48边界/16leverage。选择不读GT，小ROI补位为零权重。
2. Tri-Ownership Interaction：Self/Neighbor/BG cross-attention预测连续φN、φB和stiffness，无Direct32输出和ownership CE。φ为GT距离差先相减再裁剪到±16网格像素，未除以16。GT仅用于训练监督，Self优先，crowd忽略；空GT区域以及缺失预测邻居使用validity mask。
3. Ownership Flow：同一连续场头查询法线射线和邻近probe，Self envelope=min(valid φN,φB)的交点决定带符号位移。z>0为前景，外法线−∇z/||∇z||；约束b=δs||∇z||−z(x0)。无根/零梯度不制造flow。
4. Prototype Jacobian Compiler：固定3pass active-hinge weighted32×32 SPD求解；已满足anchor初始零force，后续pass重新激活违反项。保留ROI Gram/行权重归一化和原ridge，加预定义FP32 scale-aware numericalridge，并验证实际Cholesky info。求解具有数值ridge与logit/系数trust bound；有限pass不保证全局KKT或所有约束满足。

损失为Potential SmoothL1、同场head邻近查询的Direction cosine、真实接口法线上的Ordering，以及Compiler输出mask的task BCE+Dice。stiffness有floor，并以GT势幅度定义其监督目标。

正式fit：796张，实际匹配instance/epoch：6712；预算8epoch，seed0，每图chunk4，FP32无AMP/增广，AdamW lr=.0003/weight_decay=.0001，梯度裁剪10。实际attempt/applied：14856/14856；可训练参数：187651。dev197不用作选择，保存固定final8，不自动延长/增加seed。

训练采用class-free最大boxIoU≥.5、first64 native post-conf候选、raw index去重及每GT round-robin选最多12。部署在GT-free的first64预测域运行top2预测邻居；候选预算内无法执行的空native ROI保持c0且不补位，其他候选保持c0。官方bbox支持区固定，超框前景不可恢复。

数值诊断：20项，诊断状态passed；包含oracle扩缩/邻居/ignore/空区域、实际task-only梯度和优化器更新、FP32高尺度rank1、真实官方冻结native重放。诊断效应不是正式训练或AP效应。

## 回放、读出与配对范围

完整5000 val2017、640 square LetterBox/scaleup=false、native one2one FP32、conf=.001/max_det300；原生process_mask_native后scale_preds.byte()。COCOeval maxDets=[1,10,100]，segm读出剔除bbox字段以保留mask area语义。

官方baseline复用先前COCO输出字节，但本次必须逐5000图fresh冻结模型forward与原生decoder，证明每图检测identity与RLE完全相同。原baseline的legacy部分state digest与本次完整official state digest范围不同，不制造相等；本次两arm均绑定同一个完整冻结状态。

独立NumPy读出复核重新聚合COCO NPZ的AP/AP75/APsmall并比较bbox累计数组，核对5000图、first64/unsupported/预测邻居/no-GT域及paired点估计；不重新计算GT mask匹配、maskIoU或bootstrap置信区间。

paired诊断是same native detection index/class对应的同类非crowd GT最大boxIoU≥.5，允许多个预测匹配同一GT。baseline maskIoU≥.75算成功，掉到.75以下为damage，失败变成功为repair；它和COCO AP匹配、原始候选五状态几何不是同一指标。Image-cluster bootstrap5000/seed0仅用于paired IoU/damage；未计算AP CI。

| 配对点估计 | 值 |
|---|---:|
| matched_detection_count | 86600 |
| baseline_success_count | 36266 |
| damage_count | 447 |
| repair_count | 693 |
| mean_mask_iou_delta | +0.002082 |
| damage_rate_of_baseline_success | 0.012326 |
| repair_rate_of_baseline_failure | 0.013768 |
| baseline_success_mean_iou_delta | +0.000332 |

## 推理域与开销

| 项目 | 实测 |
|---|---:|
| selected_instances | 229596 |
| refined_instances | 228974 |
| unsupported_roi_instances | 622 |
| boundary_valid_count | 4989659 |
| root_count | 3443203 |
| invalid_boundary_root_count | 1546456 |
| neighbor_transition_count | 1354949 |
| active_anchor_count | 3123814 |
| trust_saturated_instances | 209431 |
| solver_instances | 228974 |
| solver_nonzero_cholesky_info_count | 0 |
| gram_numeric_ridge_instances | 0 |
| system_numeric_ridge_instances | 0 |
| maximum_recorded_system_condition | 305344.4375 |
| maximum_system_residual | 3.7823094316991046e-05 |
| actual_frozen_forward_seconds | 493.929 |
| extra_module_seconds | 3564.369 |
| extra_module_ms_per_image | 712.874 |
| total_elapsed_seconds | 5186.609 |

时间包含Python、adapter和数据转移，不能当成优化部署吞吐。具体numericalridge、逐pass条件数/违反次数/oscillation及trust scale保留于TRAIN_TRACE/TASK_GRADIENT_EVIDENCE和COMPILER_IMAGES。invalid_boundary_root是无可用root的边界row，不等于solver失败；非SPD/非有限solver会使Run明确失败。

## 预先锁定的科学门槛

| 门槛 | 原始读出条件 |
|---|---|
| Mask AP增加至少0.3 AP点 | 未通过 |
| AP75与APsmall不能同时下降 | 通过 |
| baseline成功实例damage比例≤1% | 未通过 |
| baseline成功实例平均IoU增量≥−.005 | 通过 |

只有以下全部必要执行/证据链通过，才把上述数值条件解释为有效正/负结果。

## 执行与证据链

| 检查 | 状态 |
|---|---|
| RUN_TRIFLOW_PREFLIGHT_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_PREFLIGHT_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_SMOKE_CACHE_S0 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_SMOKE_CACHE_S0 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_SMOKE_TRAIN_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_SMOKE_TRAIN_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_FIT_CACHE_S0 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_FIT_CACHE_S0 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_TRAIN_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_TRAIN_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_EVAL_CONTRACT_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_EVAL_CONTRACT_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_COCO5000_EVAL_S0 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_COCO5000_EVAL_S0 回传验证及原始 manifest 保留 | 通过 |
| RUN_TRIFLOW_PIPELINE_S0_R1 执行完成且 exit0 | 通过 |
| RUN_TRIFLOW_PIPELINE_S0_R1 回传验证及原始 manifest 保留 | 通过 |
| 实际环境、原始训练/验证标注及 native 输入预检 | 通过 |
| 20 项数值、task-only 梯度与真实冻结 native 重放均通过 | 通过 |
| 通过诊断的实际 source archive 与训练核心/提取器一致 | 通过 |
| 796 张正式冻结 cache 完成并绑定 receipt 字节 | 通过 |
| 逐图 FP32 cache/image/input SHA、数量与大小 inventory 一致 | 通过 |
| cache source archive、正式训练源与通过诊断源一致 | 通过 |
| 训练 first64、raw 去重、class-free boxIoU≥.5、GT round-robin≤12 | 通过 |
| Self 优先、crowd 忽略、BG 独立、N 缺失监督有 validity mask | 通过 |
| 正式训练固定8 epoch、seed0、chunk4、FP32无AMP/增广、固定AdamW | 通过 |
| 实际 native feature/hidden 维度、所有锁定默认值与 configSHA | 通过 |
| 正式训练绑定实际796 cache、原始训练标注与列表SHA | 通过 |
| 实际8轮796图/instances顺序与 applied step 计数一致 | 通过 |
| 实际更新有限、无跳过且6个模块组确实改变 | 通过 |
| 同一真实mask task-only probe经Compiler回到φ前2行与attention；无隐藏warm-up | 通过 |
| 仅训练新模块；每轮首chunk输入hash不变，cache实际逐字节验过 | 通过 |
| 唯一正式final8 head字节/大小绑定训练completion与已加载head | 通过 |
| 评估确实加载该final8 head及正式配置/cache/source链 | 通过 |
| 4图official c0 native identity/RLE与真实smoke拒绝；不是formal正例/AP/5000证明 | 通过 |
| 完整original COCO5000评估及completion->SUMMARY SHA | 通过 |
| 官方8.4.100/native one2one/FP32/640/原始mask-area AP协议 | 通过 |
| 实际evaluation/readout/core/extraction source archive及训练源一致 | 通过 |
| 部署first64预测域、top2预测邻居、无GT/Direct32；unsupported保持c0 | 通过 |
| 原baseline逐5000图fresh官方forward/native decode与identity/RLE完全相同 | 通过 |
| 两arm同一完整冻结official模型digest且predictions字节绑定 | 通过 |
| 5000个baseline逐图文件当前SHA与fresh replay链逐个绑定 | 通过 |
| 每部署instance/3pass实际SPD info0、finite数值ridge、trust≤2logit | 通过 |
| NumPy独立NPZ AP/bbox数组/paired点估计/5000域重聚合且所有引用文件SHA未过期 | 通过 |
| same native detection index / bbox / class / confidence，5000图完整配对 | 通过 |
| 评估期间learned module state不变 | 通过 |
| 所有当前必要receipt可解析，无错误伪装为缺失 | 通过 |

## Run记录与已知失败

| Run | 执行 | exit | 产物 | 回传 |
|---|---|---:|---|---|
| [RUN_TRIFLOW_COCO5000_EVAL_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_COCO5000_EVAL_S0/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_EVAL_CONTRACT_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_EVAL_CONTRACT_S0/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_EVAL_CONTRACT_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_EVAL_CONTRACT_S0_R1/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_FIT_CACHE_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_FIT_CACHE_S0/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_NUMERICAL_VERIFY_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_NUMERICAL_VERIFY_S0/REPORT.md>) | failed | 1 | available | verified |
| [RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_NUMERICAL_VERIFY_S0_R1/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_PIPELINE_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_PIPELINE_S0/REPORT.md>) | failed | 1 | available | verified |
| [RUN_TRIFLOW_PIPELINE_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_PIPELINE_S0_R1/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_PREFLIGHT_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_PREFLIGHT_S0/REPORT.md>) | failed | 1 | available | verified |
| [RUN_TRIFLOW_PREFLIGHT_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_PREFLIGHT_S0_R1/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_SMOKE_CACHE_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_SMOKE_CACHE_S0/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_SMOKE_TRAIN_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_SMOKE_TRAIN_S0/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_SMOKE_TRAIN_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_SMOKE_TRAIN_S0_R1/REPORT.md>) | completed | 0 | available | verified |
| [RUN_TRIFLOW_TRAIN_S0](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_TRAIN_S0/REPORT.md>) | failed | 4294967295 | available | verified |
| [RUN_TRIFLOW_TRAIN_S0_R1](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_TRAIN_S0_R1/REPORT.md>) | completed | 0 | available | verified |

首次preflight与numerical S0失败历史保留，不被R1成功覆写。最初anchor fixture λ·ε=1e−8在单位尺度共线FP32系统里被舍掉；R1用正式ridge，另加大尺度rank1数值ridge验证。旧Pipeline/Train S0在加强真实φ-task/attention同probe证据前保留中断/失败记录，新pipeline、smoke、contract和正式train使用实际R1 ID；它们不延长训练预算。正式核心已锁，未把diagnostic oracle或warm-up算成正式训练。

## 结论限制与未测项

本次为历史cohort、单seed、固定8轮新模块试验；与ACD的可训练范围、目标和训练预算不同，不能作单变量因果归因。最终可说明此固定部署域/预算是否达到门槛；不能据此证明论文新颖性、普遍有效或普遍无效。raw五状态/AUC/FPR域/Ucrop/AP CI及论文发表适用性保持未知。

关键产物：

- [锁定协议](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/PROTOCOL.md>)
- [三值门槛决定](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_COCO5000_EVAL_S0/DECISION.json>)
- [正式评估原始SUMMARY（若已回传）](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_COCO5000_EVAL_S0/SUMMARY.json>)
- [独立aggregate核验（若已完成）](</C:/Dpan/codexproject/paper-disc-organized/experiments/triflow_potential_coefficient_20261006/runs/RUN_TRIFLOW_COCO5000_EVAL_S0/READOUT_VERIFICATION.json>)
