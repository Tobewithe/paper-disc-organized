# RFC-v1 训练与固定checkpoint评价配置复核

2026-10-17结案。**实际配置复核通过；科学效用/机制门槛未通过。** 科学结论见[整项实验报告](REPORT.md)，配置通过不替代方法Gate。

| 范围 | 冻结值与实际证据 |
|---|---|
| 基础模型/框架 | 原yolo26m-seg.pt与Ultralytics8.4.100；权重、框架8项关键源码与baseline_profile SHA核验通过。P、c0、基础模型和Box/Class/score/排序/最终输出身份冻结。 |
| 原生推理 | imgsz640、batch16、rect=True、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300、mask_ratio4、overlap_mask=True、retina_masks=False、workers0、无augment/compile。其余实际NATIVE_ARGS也与原入口比较，不靠默认参数名推定一致。 |
| 批次/输入几何 | 新抽样后的1294评价图分布于309个原生批次；重放完整4936上下文图，保持批次成员/顺序/input_hw/ratio_pad/图像与输入张量指纹。535225条原生输出P/c0/单点F及zero-delta RLE精确重放。只为1294张图保存单独区域F sidecar，不改原缓存。 |
| A/Q训练 | 各原2000图、3轮、1500步、6000图访问，均167872参数/17张量；IMAGE_ORDERS/TRAINING_JOIN/initial.pt与G或原B逐字相同。17条梯度路径有效；独立读取checkpoint的17个AdamW参数状态step均为1500。新增评价训练步数0。 |
| 优化和loss | 原soft-IoU＋0.1保护BCE；AdamW lr=.001、betas=.9/.999、eps1e−8、wd1e−4；FP32，无AMP/调度/早停/增强；最后epoch checkpoint，无择优。A/Q只有F排列差异。 |
| 实际数值环境 | 笔记本Python3.12.3、torch2.5.1/CUDA12.4、torchvision0.20.1、NumPy2.4.3、pycocotools2.0.11；matmul highest、matmul TF32=False、cuDNN TF32=True、benchmark=False、cuDNN deterministic=False、deterministic_algorithms=False、CUBLAS_WORKSPACE_CONFIG=:4096:8。这些实际值与G一致；没有把CLI deterministic=True误当成后端实际全确定性。 |
| 允许干预 | 同层8×8区域F共享投影；A有序/Q固定SHA置换；J只置换同A权重的F。原生掩码面积≥9216才应用Δc，门外RLE逐字保留；c@P到原图的原生二值缩放/byte/RLE链路不改。 |
| 等幅机制控制 | 两组FP64全proto预测框ROI的PΔc RMS，双方只缩到较小幅度，原1e−7/1e−5容差；11172个有效匹配行、101904个门外行，全部保留。5586个门控内F均非恒定且确实被置换改变。无GT用于模型forward、置换、门控或幅度选择。 |
| 比较与分母 | N/G/A/Q在同1294图/184439输出上完整评价；J和四个等幅组在同dev512/56538输出上完整评价。T655/257有效图、全部bbox75配对2925/508有效图。GT task与同bbox检测IoU分别计算，bbox/segm重新独立匹配。 |
| 统计与损伤 | 原图IoU、图宏归约、2000次配对图bootstrap、升序image_id、FP64、default_rng1729、linear分位。两checkpoint/源码/推理合同先冻结，再一次系统随机seed抽22957成功GT中的1024条；原面板在工程恢复中逐字复用，无补样或重抽。全图预测/GT匹配，只数样本GT；整数超几何上界。 |
| 旧G核验 | S3完整预测SHA 15d8cc2884777854a7120099e4ee218b8750ee59f63d2b3fcc4571fa2a191590；完整N身份与门外RLE一致。开发T+.010024491543988425、全部配对+.0008909711434980322原端点精确复现；新随机面板损伤6/1024与旧抽样值不同，不构成配置漂移，也不覆盖旧全量90/22957 census。 |

## checkpoint、来源和真实执行

- A：RUN_RFC_TRAIN_A_59b094a35f9b4fc3a45f7a86862ec677，last SHA `1383818808b7c52594ba5dfb49ae8f4b858a900b02fb65dccd5f1c55e6bf1963`。
- Q：RUN_RFC_TRAIN_Q_ed1e5e49f60c42908d23822e5804f668，last SHA `877a3b221fb0b81fa20772495cfbade530402252202a8fff86a2cf816c058942`。
- G：RUN_SOFT_IOU_TRAIN_53afd0bb8aba428482159f5e426878eb，last SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`。
- 评价冻结：RUN_RFC_EVAL_FREEZE_1cea8787508c4d59a29c2a1dbb75c451；原生输入：RUN_RFC_EVAL_INPUT_70a101e0c33644bc97e2be00ac41bdfa；结案：RUN_RFC_EVAL_RECOVERY_e2b485f03a5f4e96a0f96ab327f4e39f。
- [训练独立核验](preflight/TRAINING_CHECKPOINT_INDEPENDENT_LOCAL.json)、[评价源码冻结](EVALUATION_SOURCE.json)、[恢复源码冻结](EVALUATION_RECOVERY_SOURCE.json)与Run内source_snapshot记录实际未提交源码。全部原训练源码保持不变。

## 工程中止和恢复边界

原冻结入口RUN_RFC_EVAL_FREEZE_dcf258fd926f4634b75574c35d06bb56把checkpoint的steps字段读成optimizer_steps，抽签/预测前失败；已修正字段读取并核训练合同，未换权重。

原评价RUN_RFC_EVAL_FULL_82c717bc94c84c87959c84ae0051d6c4在81图后触发原7GiB显存检查。旧Run未分拆保存allocated/reserved峰值，不能编造旧峰值。恢复只在每图结束后释放未用CUDA缓存，保留历史最大峰值检查；原81图**系数张量和全部预测RLE精确一致**，见[重放收据](runs/RUN_RFC_EVAL_RECOVERY_e2b485f03a5f4e96a0f96ab327f4e39f/FAILED_PREFIX_REPLAY.json)。不改变门槛或数学运算，不重抽面板。

首恢复入口在计算前发现旧失败manifest早于预测流最后flush。9份预测文件只有尾部追加，旧哈希前缀全部精确一致，其余产物不变；原manifest保留，另存[进程退出后清单及证明](preflight/FAILED_EVALUATION_POSTEXIT_MANIFEST.json)。恢复代码逐图flush，避免同一写入问题。旧源码和启动日志保留，未把工程中止算作方法阴性。

原F导出总耗时914.672+1571.125=2485.797秒<10800秒；正式基础图forward为2000+4936=6936≤7000，另列原smoke256图，总7192不混作正式预算。评价失败与恢复合计141.422+1381.797=1523.219秒<14400秒。恢复评价CUDA历史峰值allocated约1.079GiB、reserved约1.545GiB；运行内采样RSS峰值约2.150GiB。原输入重放CUDA reserved峰值约6.537GiB，也低于7GiB。F sidecar共约21.23GiB<48GiB、其它Study产物约0.556GiB<8GiB；起始RAM与磁盘预留检查通过。详细值见Run内CONFIG_AUDIT/MEMORY_LIFECYCLE，不以空闲显存代替实际峰值。

## 回传和独立复核

2041项传输文件哈希全部一致，原已完成Run不覆盖；完整新预测、系数、账本与配置/统计已回传，区域F大缓存留笔记本并以清单/SHA关联。[回传收据](preflight/TRANSFER_VERIFICATION_06.json)。两机使用同一[独立核验源码](scripts/verify_evaluation.py)，没有调用生产统计汇总函数；各自1399项原始产物哈希、26325次GT RLE配对IoU、54项统计数值精确相同、113076行幅度核验及损伤整数上界通过。见[笔记本](runs/RUN_RFC_EVAL_RECOVERY_e2b485f03a5f4e96a0f96ab327f4e39f/INDEPENDENT_LAPTOP.json)和[本机](runs/RUN_RFC_EVAL_RECOVERY_e2b485f03a5f4e96a0f96ab327f4e39f/INDEPENDENT_LOCAL.json)。

模型执行始终使用原笔记本环境。本机独立复算用Python3.12.14与单独临时目录内的NumPy2.4.3/pycocotools2.0.11，只读相同标注SHA与回传产物，没有改变笔记本配置或新增模型运行。[验证环境/源码](VERIFICATION_SOURCE.json)。官方发布AP一致性仍是历史独立未通过状态；本轮同配置N/G/A/Q比较与原生GT账本合法性通过，不改写该历史Gate。
