# CLD-v1 正式训练后配置复核

**结论：REAL/PERM正式训练配置符合启动前冻结合同；除预注册的实例对应变量外，两份合同逐字段相同。** 原生推理、Adapter训练与尚未执行的评价配置分别核验；正式训练完成不等于科学Gate通过。

| 检查项 | 实际生效与证据 |
|---|---|
| 科学及源码身份 | PROTOCOL SHA `053aaefb1cbb0fd5f87dad2dabdf9e961660201ae81bae6c8e42e67d61d6d453`；11项科学文件、36项执行文件及两个Run的源码快照逐项匹配。执行freeze SHA `56e9f26f4ebf2f56df92de068044ccaafdae60278b6b91b9a204751058fa12a8`。启动控制另冻4项，不修改模型数学。 |
| 环境与包 | 两臂一致：Python3.12.3、torch2.5.1、CUDA12.4、Ultralytics8.4.100、torchvision0.20.1、numpy2.4.3、pycocotools2.0.11、OpenCV4.13.0.92、Pillow11.1.0、PyYAML6.0.2；8项Ultralytics源码SHA与继承环境一致。 |
| 数值运行开关 | `matmul_precision=highest`、CUDA matmul TF32=false、cuDNN TF32=true、benchmark=false、cuDNN deterministic=false、deterministic_algorithms=false、warn_only=false，CUBLAS workspace `:4096:8`。这些值沿用冻结环境，没有为本次结果改成另一组“更确定”默认值。OMP=1、MKL未设置、YOLO离线及PYTHONPATH固定。 |
| 基础模型与缓存 | 固定权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。启动前2000原图/2000缓存以及原预测、教师、完整及子集两种标注均核SHA；未重新导出检测，基础模型forward=0。 |
| 原生推理配置 | 缓存来源保持imgsz640、batch16、rect=True、end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300、mask_ratio4、overlap_mask=True、retina_masks=False；逐图input_hw/ratio_pad、P/c0/F及检测行不变。**训练microbatch1不等于重新用batch1推理。** 原生输出与教师解码重放见已通过预检和smoke。 |
| 模型与初始化 | 两组均为原B结构167872参数、17项白名单。各自重新读取原initial，文件SHA `ce2de59a316eef3cea6cfe6d1763221931fd10693377c8a91ce321843604bacd`；未从smoke或soft-IoU last接续。基础模型不在optimizer中。 |
| 实际优化器 | AdamW lr=.001、betas=(.9,.999)、eps=1e−8、weight_decay=.0001；amsgrad/foreach/maximize/capturable/differentiable/fused均false。FP32、AMP关闭，无增强、scheduler、EMA、早停或选轮。独立脚本核实保存optimizer每个参数的step=1200。 |
| 框架默认键权限 | 缓存val参数中的epochs100、optimizer=auto、amp=True未用于本Adapter训练；本轮采用显式合同和保存的实际optimizer状态。不能拿未生效的val配置键解释本次训练。 |
| 固定样本及预算 | 每臂原1600拟合图×3轮=4800访问，microbatch1/累积4=1200更新。全部4800条图序与冻结IMAGE_ORDERS一致，400留出图梯度访问0。每臂2510教师×3=7530访问，176042输出×3=528126访问；没有删无教师图或只保留门控内实例。 |
| REAL/PERM的变量 | 两组合同去掉arm后完全相等。REAL保持对应；PERM按冻结同图无自配双射重排行d，再用接收实例P/c0/ROI/目标计算loss。相同初始化、图序、幅度、保护、优化器与预算。非干预参数未漂移。 |
| 方向监督与保护 | 拟合教师`(c*−c0)@P`在原proto ROI用FP64求RMS；目标转FP32，方向MSE逐实例再逐图等权。共同a=2.1651711144258443，只来自2510拟合教师。保护为全部非空输出ROI的native probability BCE×.1。全日志loss关系在原工程容差内，无NaN/Inf。 |
| 独立核验范围 | 正式Run的冻结verify_cld核序列化参数、optimizer、合同、源文件及全图序，未重复1200步梯度计算；原smoke已完成梯度/显式AdamW核验。本机不运行torch模型，仅独立核哈希、合同、完整日志及预算，不把两地核验表述为两次全训练复现。 |
| checkpoint与后续评价 | 仅两个last；[TRAINED_CHECKPOINTS](TRAINED_CHECKPOINTS.json)已固定文件/状态/合同SHA。未启动400图模型评价、未算新AP、未改原门控9216或decoder。相对N、历史G及科学对照的修复增量和损伤均未测。 |
| 官方基线状态 | 仍为native_config_scoped：原生GT账本有效、官方刊载AP parity未通过。训练配置通过不改变这两个独立状态，也不改变历史S2/S3结论。 |

## 预算与回传

本次正式训练共2400更新、9600次Adapter forward；此前工程阶段4更新、80次forward（含32次独立复算）单列，Study累计2404更新/9680次forward。基础模型forward、新oracle、留出模型评价均0。REAL/PERM runner分别247.047/299.047秒，均低于每臂2小时上限。

134项正式Run及控制产物回传，逐项大小与SHA通过；实际总大小12025592字节。源端清单SHA `4865f6cde3b931b726d1a78687fe75e1ed4ca273e3b0bf65ab23a677809e094c`在本机复核一致。清单顶层`total_bytes`因PowerShell汇总写为null，逐文件大小完整；本机按逐项独立求和，原清单保留，派生总量见[传输回执](FORMAL_TRANSFER_RECEIPT.json)。不改写原Run中启动时的`transfer_status=local_only`；当前可用性由Study及传输回执登记。

两项控制层收尾记录保留：①任务导出文本曾为UTF8字节/UTF16声明，保留原`FORMAL_TASK.xml`并另存有效`FORMAL_TASK_UTF16.xml`，见[导出修复回执](FORMAL_TASK_EXPORT_REPAIR.json)；②训练控制器成功退出后任务仍Ready，收尾显式停用，XML确认无触发器，见[设备收尾](FORMAL_REMOTE_CLOSURE.json)。两者均未改变已注册训练动作、模型代码、Run、预算或科学门槛。

证据：[训练报告](TRAINING_REPORT.md) · [原科学协议](PROTOCOL.md) · [启动复核](FORMAL_PRELAUNCH_CHECK.json) · [本机正式核验](FORMAL_VERIFICATION.json) · [完整传输清单](FORMAL_TRANSFER_MANIFEST.json) · [控制器状态](FORMAL_STATUS.json)。历史[工程阶段配置复核](CONFIGURATION_REVIEW.md)原样保留。
