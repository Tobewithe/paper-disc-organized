# 训练内可达掩码目标对照（AMT-v1）

日期：2026-10-16。研究线：原型—系数链路（`neighbor_sensitive_p3`）。
执行环境：用户已授权的笔记本 `ssh 28358lan`，仓库镜像 `D:\coco_wire\baseline_native_20261013`。
本文件在教师求解、烟雾训练和正式训练之前冻结；实际生效身份见 `FREEZE_MANIFEST.json`、每个 Run 的 `EVAL_CONTRACT.json` 和源码快照。

## 1. 主张、对照与边界

H：在原 2000 图和 3 epoch 的预算下，将一部分**可恢复失败**的不可完全表达 GT 目标，替换为同一检测、固定 P 可表达且原图质量经过核验的二值掩码目标，可使 adapter 学到更多可部署修复，同时保持原成功保护。

正面依据：S1 有限 oracle 找到 122/256 条修复；机会核对中其中 100 条未被当前 adapter 修复、仅 10 条已修复却被面积门控撤回；训练内核验显示门控内 train/dev 平均收益相近，而无门控训练内损伤仍为 394/9748。优先检验修复监督，不先扩量。竞争解释是当前输入/架构/短预算不能迁移这些修复；教师机会充分但新目标无增量时，关闭本实现，转向修复条件的可预测性，不再重复换目标或扩量。

这是假说，不是“GT 监督已被证明是瓶颈”。历史 `coefficient_oracle_alignment_20261009` 的原生分支系数方向回归阴性保留；本次不回归系数向量/余弦，不解冻基础模型。旧 val S1 的 256 条不构建教师、不选超参。可达指有限找到的**同 P 二值函数**，不指真实 GT 完全可达、全局最优或 soft-IoU 连续最优可达。

唯一训练科学变量：**目标生成策略**。继续原 soft-IoU 函数，不同时改 loss 形式、权重、样本量、匹配、架构或训练步数。旧 soft-IoU Run 是确定的复用对照，不重复花预算训练它。

## 2. 不变输入与预算

| 项 | 冻结值 |
|---|---|
| 原初始化 | `RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/initial.pt`，SHA `ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd`；不从 soft-IoU last 接续 |
| 原训练缓存 | `RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec`；TRACE SHA `fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d`；原 fit2000 和逐 epoch IMAGE_ORDERS 字节完全一致 |
| 模型 | 原 `CoefficientAdapter('B')`，167872 参数；冻结 P/c0/F、YOLO 权重、所有检测行身份/框/类别/score/顺序；不运行基础模型 |
| 训练 | seed0，3 epoch，6000 图访问，1500 AdamW 更新；microbatch1、累积4；lr.001、betas(.9,.999)、eps1e-8、wd.0001；原其余 optimizer flags，全 FP32，无 AMP/增强/调度/早停；只用最后一轮 |
| 匹配 | 原训练 class+Box50 score-order 一对一普通 GT join，14276 条、14268 个非空 ROI；不以 S1 Box75 配对替代训练 join，不删难例 |
| 原对照 | `RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb/last.pt` SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`；开发固定配对与完整输出复用 `RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0` 和面积门控 Run |
| 原生配置 | 8.4.100、原权重 SHA、原 runtime/package/source pins；imgsz640/batch16/rectTrue/end2endTrue/agnostic_nmsFalse/conf.001/iou.7/max_det300/mask_ratio4/overlap_maskTrue/retina_masksFalse；按缓存保留每图真实 geometry，不重新分批 |

教师离线求解是额外数据准备计算，必须单列；不得称为与普通 soft-IoU **总算力相同**。正式教师仅一次，CPU FP64 单线程求解，GPU 使用原 native decoder；最多 4 小时。教师烟雾为原 fit 顺序前 8 图，最多 15 分钟；正式教师仍完整求解 2000 图（烟雾成本另列，不选教师结果）。训练烟雾同 8 图、1 epoch、2 更新，最多 15 分钟；正式训练从原 initial 重新开始，最多 2 小时。独立教师重放最多 1 小时，训练独立核验最多 15 分钟。越界/数值错误停止，保留全部证据；不能换阈值、追加教师重启或训练预算。

## 3. 教师构建与身份

1. 使用原训练 join 和原生训练 GT 账本 `RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/BASELINE_GT_LEDGER.jsonl`。普通 GT 原 COCO Mask75 任务失败、该 join 同一检测原图 MaskIoU<.75、预测 proto ROI 非空，才允许求解。GT 已成功时即使训练 join 是另一检测，也不替换其监督。
2. 对每个符合条件的 `(image_id, annotation_id, output_detection_id)`，用原 `project_gt`（实际 resize/pad→nearest proto）与原半开预测框 ROI，原 `finite_coefficient_solver.DEFAULTS` 求解一次：CPU float64、zero_delta、BCE_ROI + .003/2·||Δc||²，L-BFGS lr1/max_iter120/max_eval150/history20/strong_wolfe、tolerance_grad1e-9/tolerance_change1e-12，grad∞≤1e-6 判收敛。无重启、无候选/阈值搜索。
3. 转回 FP32，在**完整同图系数矩阵**中写入该行解，用原 native decoder 得到原图 RLE；逐图 c0 零残差重放须与原 JSON 精确相同。保持原 GT 全面积 union，不裁 GT 以夸大 IoU。
4. 仅当 solver 收敛、原图 IoU 增益≥.01，且教师 proto ROI 前景非空时接受。教师 `q=1[(c* @ P)>0]` 使用与原 `loss_grids` 相同的完整 FP32 CUDA 矩阵乘法，再保存 uint8 proto 网格；不将原图 RLE 重新缩回 proto。用硬目标可沿用原 soft-IoU 公式；不用连续教师概率，以免混淆 soft-label IoU 的最优点。
5. 其余全部 join 保留原 GT，包括未收敛、无改善、空前景和原成功 GT；不删样本、不改分母、不额外加权。空 ROI 与无 join 图依原规则。教师目标 detach，原型/原始系数只读；教师仅在 loss 构建时读取，adapter forward 接口不接收 GT、教师、成功标签或系数解。
6. 每条求解与不替换理由、原生/教师 IoU、solver 终态、系数解、原图 RLE、原生面积、输出全局索引和 cache SHA 全部登记。每图教师包固定所有 join，并保存接受目标及精确 mask hash；原始全部 2000 图都保留包。训练逐图核对包 SHA、join 与输入身份。

教师放行 Gate（只决定是否值得花原训练预算，不是 S1/S2 科学通过）：全量覆盖/零残差/来源/独立重放通过；求解对象≥200、收敛比例≥95%；接受≥200且占求解对象≥20%；接受者原图平均增益≥.05；其中原生 mask 面积≥9216 的接受者≥64条且涉及≥32图。最后两项是事前最小学习信号要求，不是独立样本或统计显著性证明。只要任一机会条件不满足就不训练；数值/来源不合格为工程阻断，不当作方法失败。

## 4. 训练与成功保护

`L = 原 same-ROI soft-IoU(p, target) + 0.1 × 原全输出 baseline-probability BCE`。
目标集合 key、实例均值/图均值、梯度累积顺序均保持原样；唯一变化是第3节接受行的 target 从 GT 换成 q。目标包校验后由训练侧 `projected_targets` 包装器替换，不修改原文件；原 `train()` 循环与 soft-IoU 函数字节不变。日志增加接受目标计数及目标包 SHA，用于证明实际使用。

原成功 GT 不换目标；所有非空 ROI 输出的 .1 保护项不变。推理主要候选固定 `G`：只对**原生原图 mask 面积≥96²=9216**的行应用新 Δc，其余恢复原生 RLE。面积不用新输出重新判断，不依赖 GT。原生所有框/类/score/顺序/检测身份不变。`U`（无门控）仅描述，不能因 G 失败改选 U。

这些措施不保证零损伤；最终必须由完整同图 COCO 匹配和独立成功面板验收。教师原图改善也不保证学生可学或泛化。

## 5. S2 评价与比较（训练后，不能按训练 loss 选 checkpoint）

主要候选是新 last checkpoint 的 G，固定原 dev512、T655、全部 Box75 配对2925；分母不缩为教师接受者或门控内。全部原生同图检测混合后重新匹配。bbox/segm 独立匹配，固定 Box75 效果另算，不混为任务修复。

原三门槛不变：
- T 的图内实例均值再图等权 ΔIoU≥.005，2000次配对图像 bootstrap（numpy default_rng1729、percentile linear）95%下界>0。
- 全部 Box75 同检测 ΔIoU均值≥−.001。
- 冻结 last 和推理合同后、查看新损伤输出之前，对原生成功总体22957按原 `damage_acceptance.select_damage_panel` 一次抽取1024个GT，种子由系统随机源生成保存；完整同图预测/GT经相同 COCOeval；精确有限总体单侧95%损伤上界≤1%。不能根据结果补样；上界不通过而点估计≤1%记精度不足。

新方案优于原方案还需：同一 dev T 固定配对下 `G_new−G_softIoU` 的图像配对2000次区间下界>0。复用原 soft-IoU 面积门控固定配对文件，其开发均值+.010024491543988425是身份复核值，不作为新的门槛调节依据；两者均对原 native 分母评价。并列报告 target/overall/修复数/损伤、U/G，增量主张只使用指定 G 对照。新损伤清单上对原对照可从既有 S3 完整混合预测复核，不重复训练。

所有原生 replay/行身份、源码/包/实际参数/训练日志/checkpoint audit 必须通过；配置失败不能作科学裁决。全部原 S2 门槛及预定增量比较通过，才推荐同一个 checkpoint 的 S3全5000图；本任务流水线**停在训练及训练独立配置核验**，不自动启动 S3/S4。S3仍须≥+.3 Mask AP点、配对区间下界>0、全成功损伤≤1%、Box不变。

本 COCO 开发集多次暴露，bootstrap 是开发比较，不冒充独立确认。官方刊载52.5/44.1未复现与 native GT 账本有效是两个状态；原 native52.351430154/44.014174848身份不变，不借本实验宣称官方 parity。

## 6. 决策与交付

- 教师 Gate 失败：不启动正式训练。说明是修复机会/质量/覆盖不足还是工程阻断；方向聚焦训练 join 上真正可靠、可表达的修复目标来源，不调阈值追放行。
- 教师合格但新 G 未满足原 S2/增量要求：关闭 AMT-v1，不加 epoch/seed、不改权重/教师阈值；它削弱“该目标替换在当前输入和预算可改善修复”的预测。优先问题转为在已有固定修复标签上，现有输入能否区分应修/不应修及修复形态；方法需另有协议和支持依据。
- 全部通过：正向选择可达功能目标这一具体实现，唯一下一步为同 checkpoint 全量 S3；不直接扩量。

每个真实阶段独立 RUN_ID/run.json/原命令/环境/协议/源码/输入及产物 SHA/CONFIG_AUDIT/SUMMARY/DECISION；训练只有一次正式1500步，烟雾单列。教师目标与checkpoint大文件留笔记本并关联清单；回传轻量证据。训练结束独立解析序列、6000条日志、optimizer state、初始化/最终state和实际目标使用，不能只信runner计数。结果解释按“通俗说明→有依据的肯定研究方向→原Gate与增量→唯一下一步”，如实区分已启动、已完成、配置通过、科研通过。

用户本轮明确授权冻结协议后在笔记本启动训练，覆盖此前本线“仅建议/不启动训练”的旧执行边界；不改变历史 Run。无自动提交、推送、其它会话任务或工作台修改。
