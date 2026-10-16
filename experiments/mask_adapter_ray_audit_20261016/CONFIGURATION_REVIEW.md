# 方向与幅度核验：运行后配置复核

2026-10-16。正式Run `RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b`，smoke `RUN_ADAPTER_RAY_SMOKE_4fe9912dd6dd4422bc9f0c079f6267e3`。两次执行已结束，原始Run不回写；本文件及额外审计属于结案补充。

| 检查对象 | 冻结身份、实际执行与结果 |
|---|---|
| 协议先行 | PROTOCOL SHA `068ca36039aec19d33610909bb6328d5c4f8dc433b0315fadd28e48bd2a31c48`；18:16（Asia/Taipei）冻结，正式18:21:57启动。23项冻结源码/配置逐项一致 |
| 源版本 | `c23d8a2c43d1995979ff757fe7f6ebef5199e086`＋SOURCE_DIRTY.patch SHA `8c90a0e45b233b0dcd712f3d9bb233ef18d50ccb27c2e3fc453108dce1418171`，不是仅记录main/latest |
| 模型与checkpoint | YOLO26m-seg SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；原soft-IoU B/3epoch last SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`，前后一致 |
| 模型运行状态 | Adapter eval、phase=eval:0、requires_grad=False；状态SHA `8b07cbd54f7fda24946734017c9f0a3afd68ef6d54fef27b789504ab93bca8eb`前后相同。正式仅243次缓存Adapter前向；训练步0、基础模型forward0 |
| 配对与参照 | TARGET_PANEL SHA `170676d3017e4be7cd213ff1094ca28065727807c915b7602ecdaec48238cf31`；原256条/243图完整，C159不用于缩主分母。C不重求解，N/U/G/C源文件与历史Run身份一致 |
| 基线配置 | Ultralytics8.4.100，imgsz640/batch16/rectTrue，end2endTrue、agnostic_nmsFalse、conf.001、iou.7、max_det300、mask_ratio4、overlap_maskTrue、retina_masksFalse；实际使用原缓存的每图geometry，不重新分batch |
| 实际环境 | 笔记本YANGLEI，RTX4060 Laptop；Python3.12.3、torch2.5.1、torchvision.20.1、NumPy2.4.3、pycocotools2.0.11、CUDA12.4；8项Ultralytics源码SHA及运行前后环境一致。详细包版本、路径和环境变量见EVAL_CONTRACT |
| 数值配置 | FP32实际模型/解码，候选生成FP64；matmul highest/TF32=False；cuDNN TF32=True、benchmark=False、deterministic=False，deterministic_algorithms=False、CUBLAS :4096:8，CPU1线程。profile中的deterministic=True是原生调用参数，不能冒充实际torch全局确定性开关；本轮保持历史实际状态并核对精确重放 |
| 原生解码 | 完整N×32 GEMM、首次logit>0、预测框裁剪、二值scale_masks回原图再byte；没有把最后一步改成0.5阈值。49,976行zero-delta精确重放；N/U/C与历史掩码一致；164图Δc与S3存档逐值一致 |
| 唯一科学变量 | 每条仅alpha∈[0,16]；固定候选0/1/16＋代理最高的最多6个开区间中点。2,244个实际候选，无新增范围、重选checkpoint或训练 |
| 输出护栏 | 所有框/类别/score/检测ID/顺序不变，非目标掩码不变；实际二值转换表精确通过 |
| 数值接受 | 代理logit容差5e-4＋5e-5×abs(actual)，原图IoU差≤.001；最大logit差.0001470965、越限像素0，2,244候选原图IoU差与不同像素数均0；256条全部数值有效 |
| 统计 | 原图IoU0—1，图内实例均值再243图等权；2000次配对图像重采样，default_rng1729、linear分位数；完整分母/正参照分母均通过。Mask75不是重新COCO匹配，不是AP |
| 工程检查 | 7项合成测试通过；独立smoke4条/4图，706原生行、36候选、836事件检查通过；正式未复用smoke配对 |
| 运行后输入哈希 | 23冻结源文件＋28输入文件＋247缓存＋247图像＋准备文件和工程测试收据各1＝547项通过；含面板外4张smoke图。没有修改旧输入/结果 |
| 两机独立核验 | 笔记本和本机各核对正式5,768项原图RLE、2,244实际候选、972,446事件转换、23源码快照；独立统计/CI/Gate一致。本机Python3.12/NumPy2.3.5，不使用torch或候选搜索代码；不声称本机重演GPU算子 |
| 原始回传 | 371文件/19,649,985字节逐项SHA与长度一致，含正式Run297文件、smoke45文件及根目录材料；本机独立回执为随后新增，不篡改原ARTIFACT_MANIFEST |
| 结束状态 | 正式耗时1411.625秒，处理1304.728秒，在7200秒预算内；exit0。18:46核对正式PID13284和launcher18104均已退出，无Python任务、无Study锁 |

科学判定为`DIRECTION_PRIORITY`；这表示本次有限调幅没有兑现主要已知机会，不表示新可部署方法通过。未评价全22957原成功GT损伤、未得到新AP、未追加训练或启动S4。官方发表AP parity仍未通过，原生GT账本有效的状态独立保留。

[FREEZE_MANIFEST](FREEZE_MANIFEST.json)、[PREPARATION](PREPARATION.json)、[正式CONFIG_AUDIT](runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b/CONFIG_AUDIT.json)、[执行环境与命令](runs/RUN_ADAPTER_RAY_FULL_362251e449e3493396e69c56562af96b/run.json)、[输入复核](POST_RUN_INPUT_AUDIT.json)、[执行端原始清单](EXECUTOR_TRANSFER_MANIFEST.json)、[本机回传核验](TRANSFER_LOCAL_VERIFICATION.json)、[执行端退出状态](EXECUTOR_FINAL_STATE.json)。

后加的`audit_inputs_after_run.py`、`capture_transfer.py`只做结束后哈希/产物登记，不属于23项冻结科学源码，不参与模型输出或科学裁决；没有把结案工具回填成预注册科学方法。首次371文件回传核验早于Study状态更新；旧study.json原字节另存[EXECUTOR_STUDY_AT_TRANSFER](EXECUTOR_STUDY_AT_TRANSFER.json)，SHA与首次清单一致。所有原Run文件未改；随后Study结案状态和报告通过单独的CLOSEOUT_MANIFEST/远端核验收据同步。
