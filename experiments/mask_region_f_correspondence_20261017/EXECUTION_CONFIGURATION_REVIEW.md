# RFC 原生集成与训练启动配置复核

2026-10-17。本次复核范围为原生重放、真实 smoke 与已授权正式训练的前置；不是训练结束审计，也不是科研效用验收。正式训练状态见 `EXECUTION_STATUS.json`、各 Run 的 `run.json` 及当前总控。

| 项目 | 冻结要求与实际证据 |
|---|---|
| 环境与源码 | 笔记本 YANGLEI，原 Python3.12.3 / torch2.5.1 / CUDA12.4 / Ultralytics8.4.100；完整环境树、8个框架源码SHA与G合同精确比较通过。保留原数值开关，包括 cuDNN TF32=True、CUDA matmul TF32=False，不自行改成另一种“更确定”的配置。 |
| 来源 | 33项既有来源SHA通过；唯一文档位置映射是旧版《基线复现与配置一致性》的固定快照，不涉及计算代码或配置。原位旧文档SHA和当前快照SHA都核验。 |
| 推理预处理 | 原batch16、rect=True、imgsz640及其实际补边尺寸；完整数据集先按原生规则分组，再选原完整批次。实际参数树仅排除输出/数据/模型路径字段，其余全部比较。路径对应输入另核SHA，逐图输入tensor SHA亦精确一致。 |
| 张量与检测身份 | 输入smoke选fit8/dev8，连同完整批次上下文共256图。原P/c0/单点F/框/类别/score、raw位置、最终输出顺序、原预测行身份均精确一致；基础模型参数/缓冲/模式指纹未变。 |
| 解码与零残差 | 原native serializer全部输出字段及zero-delta RLE精确重放通过；初始化A/Q的Δc为0，掩码亦精确回到原生。不更改阈值、裁剪/缩放顺序或二值resize规则。 |
| 新增输入的唯一变化 | 原同层8×8区域F，共享旧投影，无新增参数。A保留排列，Q只打乱整通道向量的位置。两臂仍为167872参数/17张量；广播原点F时与旧G forward精确相同，同权重J=Q。16张smoke图中15张的A/Q非零输出有区别，证明干预被实现，不据此判断其科学价值。 |
| 梯度与监督 | A/Q各2步smoke，17个参数张量均有非零有限梯度；原native输入只读且无梯度。原GT配对核对通过，训练不使用面积门控筛行。 |
| 正式训练冻结 | 两臂各2000图、3轮、1500步，原IMAGE_ORDERS，原initial张量SHA `ff3107ba29c0a330d1849c8373f3c7210c6c89d0382da5e1fcec367e376fd6a2`；单seed0、微批1/累积4、FP32、无AMP/增强/调度/早停，固定last。smoke权重弃用于正式训练，仍保留其产物。 |
| 优化与loss | 原训练循环；AdamW lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=.0001，其余开关逐项比较；原同ROI soft-IoU（epsilon=1e-6）+0.1原生概率保护BCE，无loss替换。 |
| 资源 | 输入重放峰值RSS 1,763,561,472字节，CUDA allocated 3,280,199,680、reserved 5,941,231,616字节；A/Q训练smoke资源亦通过。初始空间预留经旧归档迁存后满足64GiB，新F≤48GiB、其它产物≤8GiB限制保持。 |
| 评价边界 | 当前没有科研评价。后续按原协议同时比较N/G与A/Q/J和等功能幅度控制，原512开发图及S2门槛不变；损伤面板待最终checkpoint固定后按协议一次抽取。没有S3/S4启动。 |

证据：环境预检 `RUN_RFC_PREFLIGHT_7aaf3a57b5b4448e955bc7de51a0473f`；输入smoke `RUN_RFC_INPUT_SMOKE_192922660ab94ad1a8e342aa68f71572`；训练smoke A=`RUN_RFC_SMOKE_A_133d4a85e54a4edb9fb332561b48a421`、Q=`RUN_RFC_SMOKE_Q_bd05243a99704dedbf5b4bf63bfc674d`。每项Run均保留实际合同、源码快照、CONFIG_AUDIT、SUMMARY及DECISION。

[输入重放收据](REAL_INPUT_REPLAY.json) · [两臂真实smoke收据](REAL_SMOKE_VERIFICATION.json) · [执行协议](EXECUTION_PROTOCOL.md) · [来源清单](EXECUTION_SOURCE.json) · [206项回传核验](preflight/TRANSFER_VERIFICATION_02.json) · [空间清理](STORAGE_CLEANUP.md)。

本机独立对账已通过：434项产物/源码SHA核对，A/Q原初始化、环境、优化器、监督、图序及区域F来源合同一致；256张上下文的29,101条最终输出均有精确重放记录。见[独立收据核验](preflight/INDEPENDENT_SMOKE_RECEIPT_CHECK.json)。该核验复核回传凭据，不把它描述成在本机重新运行GPU模型；16张区域F原文件保留在笔记本。

完整输入阶段也已结束：2000图、218209最终输出逐图重放通过，所有训练行都有区域F，原筛选/图序没有改变。新增区域F累计20,809,578,755字节（含smoke），小于48GiB；完整阶段峰值RSS 2,047,840,256字节、CUDA allocated 3,538,147,328 / reserved 6,568,280,064字节，均在上限内。[本机108项完整输入对账](preflight/INDEPENDENT_FULL_INPUT_RECEIPT_CHECK.json)通过。

正式A启动核验已通过：原2000图/3轮/1500步合同、原初始化文件、原IMAGE_ORDERS、优化器和环境均匹配，源码与通过的smoke相同；2026-10-17 12:51:25（UTC+8）快照已执行75步。见[启动收据](preflight/FORMAL_TRAINING_START_VERIFICATION.json)。这是训练启动审计；两臂训练结束审计和科学评价仍待完成。
