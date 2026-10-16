# CET-v1 配置、身份与回传复核

结论：科学执行配置核验、原生/教师端点重放、方向/幅度重构、两机统计及166项执行端文件回传均通过。科研门槛是`INCONCLUSIVE_NOT_READY`，与工程通过分开。调度器存在已处理且保留证据的启动/收尾异常，不能把任务容器的退出码1冒称完整调度成功；三次科学Run及三个独立核验进程均实际完成。

## 冻结与实际生效配置

| 项目 | 实际值/处理 |
|---|---|
| 执行机/解释器 | YANGLEI；`C:/Users/28358/anaconda3/envs/pytorch/python.exe`；Python3.12.3 |
| 基础模型 | yolo26m-seg.pt，SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；检查哈希但没有模型forward |
| 框架 | Ultralytics8.4.100、Torch2.5.1/CUDA12.4、NumPy2.4.3、pycocotools2.0.11；8个框架源码模块SHA与原合同逐项相同 |
| 缓存原始推理 | batch=16、imgsz=640、rect=True、device=0、workers=0、conf=0.001、max_det=300、end2end=True、agnostic_nms=False、augment=False、retina_masks=False；完整native_args前后比对 |
| 缓存形状/批次 | 直接读取已哈希的P/c0/boxes_input/native_decode_hw/original_hw/ratio_pad；不重新组批次或图像预处理 |
| 精度 | 系数及decoder FP32；功能RMS和统计归约FP64；matmul_precision=highest，CUDA matmul TF32=False |
| 其余运行开关 | cuDNN TF32=True、benchmark=False、deterministic=False、deterministic_algorithms=False；与继承实际合同一致。原协议“TF32关闭”是简写，准确范围是本轮矩阵乘法；本轮没有卷积网络forward。未为追求一致性更改任何实际开关 |
| 配置中非执行项 | native_args仍有amp=True、mask_ratio=4、overlap_mask=True、epochs=100等框架命名空间值；本次没有训练/AMP上下文，也未用这些值重新生成GT掩码。GT来自哈希固定的COCO标注并按原图RLE核验；不能把这些字段当作本轮执行训练参数 |
| Decoder | 原生process_mask_native：P×c→输入尺寸scale_masks→0阈值二值化→预测框crop；之后按缓存ratio_pad恢复原图并byte。没有替换成“原图sigmoid≥0.5” |
| 数据 | 原2000图中固定1600/400；教师覆盖2510条/867拟合图、581条/208留出图；fit/heldout图像交集0 |
| 干预 | 仅压力重放系数；P、c0、Box/Class/score/输出顺序与身份、教师、缓存不修改。无训练、无新Oracle、无基础模型或Adapter forward |
| 强度 | 只从拟合误差估计，唯一RMS=2.020336759149976；FULL最大实际匹配相对误差5.680194059554111e−7，低于冻结5e−5 |
| 统计 | 全581条/208图；重复→图内→图间等权；PCG64 seed20261017，2000次整图配对bootstrap；原生、G及两个压力主对照均按同一面板 |

原始实际值见[FIT合同](runs/RUN_CET_FIT_82cb342bda694f4da095bfcb83c9980e/EVAL_CONTRACT.json)和[FULL合同](runs/RUN_CET_FULL_3d2de9164a2f434eba4ff3a756690c45/EVAL_CONTRACT.json)。`native_args.deterministic=True`这个命名空间值与运行时PyTorch确定性开关不是一回事，二者均照实保留；精确重放检验负责确认本次执行一致性。

## 输入、运行和独立复核

- [EXECUTION_FREEZE](EXECUTION_FREEZE.json)在首次计算Run前冻结22项科学源码/配置依赖，协议SHA为`c1d16d44177f748ab1484afa440fc308b00dd857f2a90eb5dcdac7d2f1a36f11`。所有Run前后检查源码、权重、原始/子集标注、原生预测、G预测及使用的缓存哈希。
- FIT独立复核重取2510条c0/c*并验证误差、图像权重、均值、协方差和功能RMS；误差估计使用留出行0。误差库与强度在smoke前锁定，FULL合同再记录其SHA。
- FULL的33584行原生zero-delta精确重放通过；581条教师RLE与原教师、N/G最终行非mask身份一致。
- 笔记本独立进程重新生成所有供体/置换/符号及33698组FP32压力系数，重放全部33698个压力RLE，核对实际幅度；无抽查替代全量。
- 两机分别用独立压缩RLE解析和交并算法核对35441个锚点/压力测量；重算23项比较、2000个bootstrap索引和冻结判定，数值差≤1e−12。本机NumPy2.3.5仅用于独立统计，不替换笔记本的模型执行环境。
- FIT/smoke/FULL耗时54.438/40.625/135.391秒，分别低于1800/600/3600秒预算；三个笔记本独立核验均通过。两次smoke图在FULL中按冻结分母保留，不因其结果选图。
- [传输清单](TRANSFER_MANIFEST.json)及[本地核验](LOCAL_TRANSFER_VERIFICATION.json)确认166项执行端文件SHA相同；归档SHA `1452c80ae00c2f49cb1e81a76c19dd681c030f0cd8169bd96da14f19ec5822ec`。执行端初始Study元数据另存[EXECUTOR_STUDY_AT_TRANSFER](EXECUTOR_STUDY_AT_TRANSFER.json)，不覆盖本地已更新登记。

## 保留的工程异常与最终资源状态

1. 调度器第一次无法找到裸`powershell.exe`，未产生Run；保存[LAUNCH_RECOVERY](LAUNCH_RECOVERY.json)与原任务XML，改为同机完整解释器路径。
2. 六项合成测试已通过，但首版控制器读到空ExitCode误判失败。保存[AUTONOMOUS_STATUS](AUTONOMOUS_STATUS.json)、原测试日志和原冻结launcher；通过[独立启动修复记录](EXECUTION_LAUNCH_V2.json)固定新launcher，提前缓存进程Handle、等待后Refresh并显式检查退出码。此时仍无科学Run，科学协议/代码/输入均未改动。
3. V2控制器记录了全部阶段成功和`completed_and_independently_verified`，但最终停用回执未写入，任务容器LastTaskResult为1。此收尾Shell异常的具体原因未定位；没有重跑或修改科学结果。经独立只读检查确认无本Study进程/锁后，只停用本Study任务，保存[POST_RUN_TASK_AUDIT](POST_RUN_TASK_AUDIT.json)：任务Disabled、剩余进程0、锁0。其它会话任务未修改。

工程恢复只处理任务调度与状态记录，没有改变假说、样本、强度、随机数、压力方向、门槛或评价分母。未启动原型/系数训练、S4或后继实验。
