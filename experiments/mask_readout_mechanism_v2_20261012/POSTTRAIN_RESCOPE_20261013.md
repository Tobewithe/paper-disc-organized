# 后训练主线重定向：冻结 YOLO26-Seg 的 Prototype→Coefficient 实例恢复

日期：2026-10-13（Asia/Taipei）。状态：**替换尚未冻结的旧 E0-TAL 前置/五臂 20k 草案的执行方向，planned_not_executed**。本文件为该 Study 当前唯一可执行*规划入口*；原 `PROTOCOL.md`、`GATES_SPEC.json`、此前 Run 和原论文证据仍保留为历史草案与证据，不擦改原预登记结果。未完成 G0 与实际批准前**不启动新训练**。

**执行细化：** [EXECUTION_CONTRACT.md](EXECUTION_CONTRACT.md) 与本文共同构成待冻结协议，统一目标集合、阶段映射、特征桥接、统计单位和冻结缺项。2026-10-13 本次修订仅针对未执行草案；既有 Run 不变。

## 唯一研究命题

> 对已经训练好的 `yolo26m-seg.pt`，**固定检测、类别、置信度、最终检测记录、基础 Prototype P、原生系数 c0 和其余所有预训练权重**，只训练一个新增的、无 GT 推理的实例条件读出器，输出 `delta_c`，能否把**正常最终输出**中“框已正确定位、Mask 不达标”的实例稳定救回来，同时保护已经成功的实例？

这不是研究官方预训练怎么分配 TAL，更不是修正 raw 统计后转向 TAL。**前期 GT 失败重基线只是评价工具链修复；Prototype→Coefficient 的后训练纠错始终是方法主线。**

模型输出路径：`z_i(u)=Σ_{k=1}^{K} P_k(u)c_{0,i,k}`；后训练候选方法 `c'_i=c_{0,i}+delta_c_theta(F_i,P, c_{0,i}, Box_i, z_i, score_i, class_i)`，`z'_i(u)=Σ_k P_k(u)c'_{i,k}`，仍由该检测的**原生预测框**按本 Run 声明的 decoder 生成 Mask。`K`/proto 实际输出形状从 v8.4.100 模型读取，不能仅凭口头默认 32。`delta_c=0` 必须严格等于基线二值 Mask/RLE。GT 不进入推理输入，不在推理时选预测或控制纠错。新增模块可以训练，但基础官方权重全部冻结且保留字节级校验。

## 不做什么（并解释旧证据为何不能一刀切）

- **TAL/训练历史：不设 Gate，不重建历史动态分配，不读取 TAL 作为后训练候选集合。** 后训练建立标签只需要对**已产生的实际推理检测**使用 GT 作离线正确类别/BoxIoU 配对。该训练配对不是 TAL、不是推理时 GT oracle、也不是 COCOeval 的 segm 任务成功标签。
- **不重做 30,828,840 个 raw；不重新预训练整个模型；不重开已关闭的共享仿射、PCDCR、7H、BGCR、ACD oracle 方向、直接空间 teacher 或固定信赖域原配置。** 若以后提出与这些相同的操作须先做代码/训练总体/评价对象差异审计。
- 历史固定 P 的 4,843/6,456（75.02%）有限 oracle 是 `argmax-correct GT 辅助原始候选` 的特定 ROI/解码下的**存在性证据**，**不是**最终输出 Mask75 任务失败救回率；后训练前先补同一输出身份的可恢复性证据。
- 旧 ACD 在单 seed/三轮系数分支微调下未达 Mask AP 提升门槛，且 oracle 方向未相对普通 BCE 提升并存在损伤；旧原型 P-only / P+c 同短预算也无净收益。它们关闭**原配置**，不是宣称所有后训练的附加条件读出都不可学。
- 旧 `mask_readout_mechanism_v2_20261012/PROTOCOL.md` 的“E0 官方 TAL 共享仿射审计未通过则不得训练本 Study”、“五臂×两种子×20k/8epoch 直接启动”与本次用户明确主线收敛冲突，**不作为本 Study 的当前启动条件**；旧 E0 如需完成仅是独立历史论文论断审计，不能阻塞后训练机制 Gate。本记录不改写 E0 旧实验的预注册门槛或结果。

## 冻结对象与训练/验证集合

模型：Ultralytics v8.4.100 的 `yolo26m-seg.pt`，官方 SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。O2O 模式必须实测 `head.end2end=True`，`head.agnostic_nms=False`，`head.max_det=300`，`conf=0.001`；因 Head 二轮 Top-K 最后产生的是 `raw×class` 检测行，同 raw 多类别不能聚合为单个 output id。矩形/方形输入、实际 `input_hw`、batch、FP32、COCO 映射、二值 Mask decoder、保存预测的 `prediction_ordinal` 都由原生输出协议固定。

训练前先复用 `../official_gt_failure_rebaseline_20261013/PROTOCOL.md` 正式的原生输出与 `PER_GT_FAILURE` 账本；该前序 Study 的官网发表 AP parity 和该配置账本有效性分开记账：同配置输出和独立 COCOeval 可核验才能使用此输出域；如果对官网 52.5/44.1 未对齐，不冒称严格官网基线，但**不以发表 AP parity 未过无条件否决所有后训练研究**。任何基线—方法差必须用**同一冻结原生输出集/评价器/图像清单**配对计算。

训练图片来自 train2017，按 `image_id` 固定 fit/dev 与旧拟合/方法选择清单的排除关系；val2017 已多次用于项目方法探索，只作研发验证，不得称盲测。目标新方法的独立确认需另有明确来源、未参与选择的图片/场景/视频分组。基础 COCO 预训练使用过 train2017，不能把后训练未见与基础预训练未见混淆。所有图、GT、checkpoint、来源 Run、预测 JSON 和 train/dev ID 清单锁 SHA。

## G0：只读确认正常输出合同（不进行 TAL）

- 从真实 `model.val`/等价**经验证原生**推理导出中选定检测集合 `D(x)`；按 `(image_id, output_detection_id, source_prediction_index, raw_id_if_traced, class_id, score, bbox)` 保存身份。即使同 raw 的两个类别行共用 P/系数，也是两个最终检测行。
- 检查 zero-delta 与基线 Mask 原图字节一致；预测 Box/Class/Score 和顺序完全不变；实际输入 H×W、Head 属性、Mask proto 及 native decoder 确定。使用官方 COCOeval 的 `segm75(gt)` 标签；Box75 与 Mask75 两次官方匹配可以属于**不同检测**，绝不拿两者拼接同一 raw 的质量。
- 为机制关联可固定 `d_b` 为**官方 Box75 匹配的单条最终检测**，并评估它自身的 MaskIoU；不属于 Box75 成功的 GT 单独记账，不能在本 coefficient-only 路线里许诺通过改 c 救回 Box 任务失败。
- 如果没有合法原生 GT 账本或无法保留 final prediction 身份，停止于 `blocked_G0`。原生 AP parity 失败单独记录，不偷换成标签无效。

## G1：固定最终检测的 finite coefficient oracle（只读诊断；不训练）

- 只在**已经属于冻结输出集合的、正确类别、Box75 官方配对检测 `d_b`** 上，在它自己的 `P, c0, predicted_box, input_hw, decoder` 下做每实例 `delta_c` 的 GT 辅助拟合。只对 baseline 官方 segm@.75 任务失败、且 `MaskIoU(d_b,g)<.75` 的目标集合 T 计算该有限 oracle 能否达到 Mask75；对 baseline 正确组检查其他 oracle 操作不意外改变身份。
- `P/Box/Class/Score/Rank/Decoder` 全固定。把 `P·(c0+delta_c)` 转成掩码时使用统一冻结的 native 真实解码和原图 GT；ROI 定义、正则化/步数/收敛标准在看 oracle 结果前锁定。优先复用 2026-10-10 的有限 oracle 数值代码，先做到同一候选与基线重放。
- **严格措辞：** 这只是给定损失与优化器**找到的修复机会下界**，不是“原型表示的最大表达能力”或无法恢复证书。只有 G1 说明同一最终检测存在充分、非极少数的可实现修复机会，才允许着手训练 c-only adapter。本草案 Gate 采用“在不少于 200 条有效 baseline 失败且 Box75 达标的不同 GT 中，有限 oracle 能修复至少 20%”，并同时记录分面积层/距离 0.75 的分布；若门槛不过，**结束本 c-only 输出纠错配置**，不对 GT 调参追阳性，后续表示增强另立独立假说而非直接宣称 P 不可表达。

## G2：只训后训练增量读出器（候选域必须与真实推理一致）

- **所有官方预训练权重冻结**（Backbone、Neck、Box/Class Head、原生 Mask Head、Proto26、BN 参数及 running state）；新模块按最终输出检测 `d_i` 的真实可用特征做实例条件 `delta_c`，**不更新 c0 原生头**。方法初始化 `delta_c=0`，推理只改变 mask，类别/分数/框/行数/顺序保持不变。`F_i` 必须是本次真实网络中间特征和由预测框映射出的 ROI，不能以 RGB 平均池化或伪造候选通过 smoke。
- **训练标签来自正常输出域而非 TAL：** 对 train2017 真实推理输出按正确类别、Score 顺序、BoxIoU≥.50 确定性一对一关联普通 GT，匹配**只用预测框、类别、GT Box**，不以所要改进的预测 MaskIoU 选训练候选。crowd/ignore 有专门排除规则；重复类别输出各自保留最终行身份。关联用于提供标签与 Mask Loss，**不进入 adapter 的 forward 或推理 gate**。匹配失败的检测不凭空标成“空掩码 GT”，先按 identity-preserve/ignore 策略处理，防止利用评分信号作弊。所有匹配门槛、监督权重和样本清单冻结。
- 训练需覆盖 baseline 成功与失败的目标，并使用适当的 Identity/成功保护约束，防止只救失败组却大面积误伤；绝不在推理时按 GT 失败标签开启模块。损失先只用标准 masked BCE+冻结基线保持项，匹配总监督预算的对照；不再添加新的 oracle direction teacher、TAL 辅助 loss 或多个动态 gate。
- **最小初筛**：来自 train2017 的固定 2000 张 fit；提前冻结 512 张 val2017 效果开发图，仅在基线标签计算后允许按固定 hash 规则补足目标失败样本（不查看方法输出）；损伤另按 [DAMAGE_ACCEPTANCE](../../shared/evaluation/DAMAGE_ACCEPTANCE.md) 从全体基线成功 GT 无放回抽样 n=min(1024,|S|)，训练前固定预算、B 最后 checkpoint 冻结后且方法评价前抽签；单 seed、固定短预算、零 delta 控制和同参数量的 P/z0 联合空间置换对照。与旧 ACD/box-guided residual/ROI readout 研究对照时必须对齐**训练对象为正常最终输出**、网络输入、计算预算和 decoder，否则旧阴性只作历史参考，不能冒充公平 A/B。
- **G2 放行标准（正式运行前连同随机种子、优化配置冻结）：** 原 Box75、Mask75 失败目标组的 image-macro MaskIoU 提升≥+0.5 个百分点且图像簇 bootstrap 95% 下界>0；原成功组 Mask75→失败比例的单侧95%上界≤1%；基线 bbox75 固定配对的全部普通 GT 的 image-macro IoU 不低于原模型−0.1 个百分点；对象、分母与 CI 按 EXECUTION_CONTRACT 冻结；GT 监督项与基线保持项在每图分别按各自合法实例数归一化，再按图平均，公式/实现见 MECHANISM_AND_DEFAULTS。未达门槛直接停当前形式，不追加层数、epoch、门控扫描。

## G3：唯一必要的正式验收（G2 通过才增加规模）

- 先在固定 5000 图或合规数据域对 **A=冻结原生输出** 和 **B=本后训练 Adapter** 用相同前向输出身份、原图 Mask decoder、COCOeval 计算全量 `Mask AP@[.5:.95]`、AP75、小目标 AP 与原成功损伤；`Box AP` 应逐值不变，可作为输出身份和类别/框冻结护栏。
- 正式 Gate：在前置冻结的全量评价协议中，相对同协议 A 的 Mask AP 至少 **+0.3 AP 点**，且 paired-image 重采样的双侧95% ΔAP 区间下界>0、任务损伤率单侧95%上界≤1%（有限总体全量验收时退化为精确损伤比例；统计细则见 EXECUTION_CONTRACT / DAMAGE_ACCEPTANCE），才能作为有实用意义的“后训练救援”方法。若只在开发子集有修复但全量不增益，记录为局部有效，当前方法不晋级。
- 只有 G2 初筛和 G3 对同一 pilot checkpoint 的全量效用验收均成立后，才依据 `project/科研项目通用规范/训练规模与结论边界.md` 增加到 20000 张 train2017 及有预算的多 seed 确认；不一开始就启动旧 E0 路由的 5 臂×2 seed×8 epoch。独立新图像/场景确认不可由反复探索过的 COCO val2017 冒充。
- 若 G1 显示大部分固定最终检测没有找到有质量的 c-only 修复，下一条仍属于“后训练”的科研方向可以是**冻结 P，额外学习轻量实例局部空间残差 Q**；须以已有 v2 的 `R vs R_parallel` 同参数容量/`span([P,1])` 控制为基础，另立预注册机制命题，不能把它当 c-only 路线的秘密额外自由度，也不能重开旧 TAL E0。

## 最小交付 / 停止判据

仅需 `G0_OUTPUT_IDENTITY.json`、`G1_FIXED_OUTPUT_ORACLE.jsonl`、`G1_SUMMARY.json`、`TRAIN_JOIN_CONTRACT.json`、`G2_TRAIN_RUN`、`G2_CANDIDATE_EFFECTS.jsonl`、`G3_FULL_COCO_SUMMARY.json`、`DECISION.json` 与核验日志（分阶段产生，未执行的不预填）。重用本地现有输出/RLE/raw 缓存，只在输入/decoder/候选不兼容时按最小需要重放；不上传原始 COCO 图、权重或大型 .npz。

**立即下一动作只有一个**：完成/获取 `official_gt_failure_rebaseline_20261013` 的真实正常输出 GT 账本，然后对其中 **Box75 匹配、Mask75 失败的同一输出检测** 做 G1 的有限系数可恢复性审计。**没有 G1 的正式输出身份修复证据前，不批准任何新后训练。**

