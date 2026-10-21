# YB2结案配置与产物复核

**结论：配置、执行、回传与本机独立复核通过。** 主效应实用门槛未通过是科学结果，不是配置失败。未训练、未改变已冻结协议、未追加推理或阈值选择。

| 核对项 | 实际证据 |
|---|---|
| Slurm与预算 | Job164，1×RTX A6000、4 CPU、16GiB；2026-10-20 17:14:45—17:38:28（台北时间），23:44，COMPLETED/0:0，Restarts=0。smoke16帧、full576帧，训练0步。 |
| 协议/源码冻结 | PROTOCOL SHA `b20d158036b716eeac0aab76a9e8b5e796f769a6d33e07f5bb0adba9e89d0069`；FREEZE SHA `ae51089783babd047cc2444e53b8f6f77df024eaccf57cd0cad35ad48ef828ab`。19个冻结文件，本机SHA全部一致；20项外部输入服务器运行前后均复核。后添的报告/审计脚本不修改冻结执行源码。 |
| 权重与框架 | 权重SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；Ultralytics8.4.100、torch2.5.1+cu124；八项Ultralytics源码pin与V1一致。 |
| 完整实际参数 | full、smoke与V1所有111项NATIVE_ARGS完全相同，不仅是简写配置里的参数。全部profile native_args和runtime_policy精确匹配。 |
| 推理关键项 | FP32、O2O、imgsz640、batch16、rect=True、conf=.001、iou=.7、max_det300、mask_ratio4、overlap_mask=True、retina_masks=False。矩阵TF32关闭、cuDNN TF32开启等实际flags沿用原profile；不把设置中的deterministic=True误写成底层所有确定性开关开启。 |
| 输入与身份 | 全576图、36个完整16图batch；实际416×672、原图384×640、padding(16,16)。保存逐图输入/P/c0/框/分数/类别及三层F哈希、raw身份和输出序列。 |
| 原生重放 | 14,704条正常原生输出，全部zero-delta RLE精确一致；用于图表的score>.25是冻结离线操作点，没有改变>.001原始导出。 |
| 几何GT | Blender4.2.1 LTS；73,728次场景API/BVH射线一致；1,728项居中投影/尺寸检查，最大偏差0.00007266090028679173px<0.001。中心amodal逐像素一致，无目标截断，遮挡条件满足预设。 |
| 本机完整回传 | full清单1,303项、340,166,285字节全部SHA一致，包含RGB、NPZ、Blender场景、RLE、账本及视频。整包SHA `9ea8acfdf2624d84694b331591d53d943f221463c75bce1027a19ae7ddf6dc2a`与服务器一致，覆盖run.json及manifest本身。smoke另83项、22,039,603字节通过。 |
| 冻结脚本跨机复算 | 原样运行scripts/verify.py：1,728条GT观测、432条配对、1,628个RLE/GT IoU、43,201个表面点命中、1,728项投影及全部时序汇总通过。没有新模型前向。 |
| 拓扑独立实现 | 本机OpenCV连通域对照服务器SciPy，16,280个字段精确一致，包含实质分裂/混入、漏部件、实例内误连和额外岛。 |
| 复算环境边界 | 本机Python3.12.14、NumPy2.3.5、OpenCV4.13.0、pycocotools2.0.11；与模型端NumPy2.4.3不同，仅用于读取已冻结产物及独立复算，不执行模型、不改变模型环境。所有要求的数值检查通过，原服务器结果保留。 |
| 展示视频 | 两个H.264视频各144帧、12fps、12秒，按三个seed顺序串接既有同步对照。原mp4v视频保留；转码命令/源SHA/输出SHA见videos/DERIVATION.json。未改变科研RGB/掩码/指标。 |

计分边界：中心主Gate使用144共同可见观测；全动物432观测是次指标，未替换主分母。三个适用性Gate通过，但+7.6389/+9.7222个百分点均未达到预设10点。原生压力损伤不叫新方法收益，G/LCF未评价。

复核文件：[本机冻结脚本复算](LOCAL_FULL_VERIFICATION.json)、[完整配置/传输/拓扑复核](LOCAL_CONFIGURATION_AND_TRANSFER.json)、[smoke传输](SMOKE_TRANSFER_VERIFICATION.json)、[最终数字](FINAL_NUMBERS.json)、[视频转码](videos/DERIVATION.json)。
