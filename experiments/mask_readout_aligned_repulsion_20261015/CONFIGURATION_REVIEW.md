# E4实际配置核对

本文是结果证据说明，不替代已冻结PROTOCOL/EVAL_CONTRACT。原生配置来源为 `paper/manuscripts/neighbor_sensitive_p3/baseline_profile.json`，SHA `ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45`。每个Run保存启动时合同、源码和环境；完结的Run另有 `CONFIG_AUDIT.json`。

| 层级 | 实际采用及比较边界 |
|---|---|
| 基础模型 | 官方yolo26m-seg，权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；参数/P/c0/F完全冻结，本轮不重新forward基础模型 |
| 框架和环境 | 笔记本YANGLEI，Ultralytics8.4.100源码pin、torch2.5.1/CUDA12.4、torchvision0.20.1、Python3.12.3、numpy2.4.3、pycocotools2.0.11，完整包版本在合同内 |
| 原生缓存几何 | 原batch16、rect=True、imgsz640；实际input_hw/resize/pad逐图读取缓存，不把imgsz640错误理解为所有输入恰为640×640 |
| 原生检测 | end2end=True、agnostic_nms=False、conf.001、iou.7、max_det300、single_cls=False、classes=None、augment=False；同一最终输出ID/框/类/score/顺序 |
| 掩码和解码 | 原生profile的mask_ratio4、overlap_mask=True、retina_masks=False不变；所有方法沿用冻结native decoder。adapter自己的GT损失尺度由新协议显式规定，不暗中套用框架默认训练mask_ratio |
| 数值执行 | FP32、matmul highest、CUDA matmul TF32=False、cuDNN TF32=True、benchmark=False、cuDNN deterministic=False、deterministic_algorithms=False、CUBLAS=:4096:8、OMP_NUM_THREADS=1 |
| 参数与训练 | 原B adapter167872参数；两臂各从原initial.pt重新加载、seed0、原2000图同顺序/同GT join，3轮6000图访问/1500步；图微批1累积4、AdamW lr.001/betas(.9,.999)/eps1e−8/wd.0001，其余冻结布尔选项False；无AMP/增强/scheduler/early-stop，仅last |
| 唯一实验差别 | R把GT BCE移到native输入解码尺度，保持项仍为原proto BCE×.1；REP在R上增加固定GT背景邻居排斥×.1。R/REP训练合同仅experiment_arm、repulsion_weight、changes描述不同 |
| 训练配对 | 原class/Box50/score-order ordinary GT一对一join；不是评价时的Box75配对。原join和IMAGE_ORDERS字节不变 |
| 主评价 | 原1314图185621输出、dev512图2925固定Box75配对、T655/257有效图、总体508有效图、原成功1024/N22957；bbox与segm匹配独立 |
| 统计与门槛 | 原2000次paired-image bootstrap、seed1729、linear分位数；T均值≥.005且下界>0、总体均值≥−.001、精确超几何单侧名义95%损伤上界≤1%；旧开发面板不称独立确认 |

原生参数里的 `deterministic=True` 与实际torch运行开关是不同层级，profile已分别记录；本次冻结并复核的是上述实际运行状态，没有后台开启新的确定性算法。也不以参数名存在就宣称重新训练会跨设备逐位一致。

训练后逐tensor/优化器/逐图日志实质审计通过，跨机器描述性轮均值末位差异及原回执断言失败保留在 [训练审计说明](TRAINING_AUDIT_NOTES.md)。这些差异未改变checkpoint或科学指标。实际训练开销见 `AUDIT_TRAINING_BUDGET_LEDGER.json`，工程恢复没有被算成额外科学训练预算或隐藏掉。

推理时不读取GT或训练join，只应用adapter残差。E4日志中的邻居占位字段不能当邻居覆盖统计，见 [字段说明](OUTPUT_FIELD_SEMANTICS.md)。

配置一致性不等于官方发布AP已复现。本地原生5000图基线仍为Box/Mask AP 52.351430/44.014175，官方公布52.5/44.1的parity状态仍未通过；本轮只在既定 `native_config_scoped` 范围比较。同子集辅助AP不得与官网5000图数值串表。
