# S3运行后配置复核

2026-10-15。主Run `RUN_S3_FULL_670fffd010664a648e096572fc7c476d`。**结论：在已冻结的本地原生配置范围内通过；S3增益幅度未达标不归因于本轮已检查项的配置漂移。** 官方发布AP parity仍是独立未通过状态。

| 项目 | 冻结与实际证据 | 核验 |
|---|---|---|
| 基础权重 | yolo26m-seg.pt，SHA `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`；不重新前向 | 前后输入审计通过 |
| Adapter | 原B网络、原2000图/3轮/1500步soft-IoU last.pt，SHA `15e6a0b3d6c9e74c05548bea42f654bde8d30364fe9aca030d59fab6da7751ce`，eval:0 | 同权重；独立8图Δc bit-exact |
| 框架/环境 | Ultralytics8.4.100隔离overlay、Python3.12.3、torch2.5.1/CUDA12.4、torchvision0.20.1、NumPy2.4.3 | smoke/full环境字段与原soft-IoU评价合同逐字段相同；关键框架源SHA相同 |
| 其它库 | pycocotools2.0.11、OpenCV4.13.0.92、Pillow11.1.0、PyYAML6.0.2；profile另锁faster-coco-eval1.8.0 | 主评价为pycocotools，不把另一后端混入本次统计；外部评价源码路径/哈希审计通过 |
| 输入几何 | imgsz640、batch16、rect=True所产生的每图原始H×W、ratio_pad、P/c0/F缓存 | 原样复用；未重新分批或改成640方形；5000原图及5000缓存准备阶段逐SHA核验 |
| 输出语义 | end2end=True、agnostic_nms=False、conf=.001、iou=.7、max_det300；classes=None、single_cls=False | 沿用S0实际最终行，不重做Top-K/NMS或过滤；全部539128行完整保留 |
| 掩码设置 | mask_ratio4、overlap_mask=True、retina_masks=False；原生decoder、裁剪/缩放/二值化 | 不从默认值另起一套；全539128行zero-delta RLE exact；未选行完整JSON exact |
| 数值 | FP32/highest；matmul TF32=False、cuDNN TF32=True；benchmark=False、cuDNN deterministic=False、deterministic_algorithms=False、warn_only=False；CUBLAS :4096:8 | 与原实际runtime一致；不将profile的deterministic=True误当作运行时torch开关True |
| 其它原生参数 | workers0、split=val、augment=False、compile=False、dnn=False、quantize=None、seed0 | profile SHA `ae8133e371d121c642e0a32070b3743f6f7657ded587d6ab0491456329198e45`一致；本轮不重新运行原生验证pipeline |
| 数据与GT | val2017全5000图；GT SHA `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`；36335普通GT/446 crowd | 3张空输出图也在分母；GT仅供评价 |
| 唯一方法差异 | 原生原图二值RLE面积≥9216时应用原Δc，否则原生RLE | 纯Python面积独立核验539128行；55552选中、55544改变；旧185621行严格复用 |
| COCO口径 | bbox/segm独立匹配、useCats1、IoU .50:.05:.95、maxDets[1,10,100]、标准areaRng、完整bbox+segm JSON的loadRes bbox面积策略 | 全量原生24指标和匹配对S0；方法bbox全部指标和匹配不变 |
| AP区间 | 全5000图配对抽样2000次，seed1729，PCG64，linear分位数；重复图片唯一ID，每次原COCOeval.accumulate | 真实重复图片两臂与物理复制评价逐元素等价；两机独立统计复核通过 |
| 损伤 | 原生segm75成功总体22957；完整重新匹配后90个损伤 | census精确0.3920372871%；没有面积筛分母或套用旧1024样本上界 |
| 预算 | 新训练0，基础模型前向0；formal wall11150.969s<21600s，峰值RSS约2.536GiB<7GiB | 未触发资源停止；没有S4或下一轮任务 |

准备收据记录27项原来源输入；恢复V2另锁3项旧失败证据，共30项，数量差异是已登记的来源增补。23个冻结执行/合同文件在本机当前源、smoke与full的源码清单和快照间逐SHA一致；PROTOCOL及EVAL_CONTRACT SHA与run.json一致。本机统计核验使用Python3.10.19，不参与方法推理或主评价；三次独立AP复算最大误差1.4211e−14点，符合1e−10冻结容差。

原smoke的ID步长失败保留。恢复将复制步长改为10^12，并拒绝生成≥2^53的ID；实际COCO超大crowd ID及5000次复制范围均被覆盖。科学配置未改变，旧/新smoke输出预测SHA相同。此修正只解决评价工程的编号表示问题，不用于调高结果。

证据：[正式配置审计](runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d/CONFIG_AUDIT.json)、[正式合同](runs/RUN_S3_FULL_670fffd010664a648e096572fc7c476d/EVAL_CONTRACT.json)、[冻结源码清单](FREEZE_MANIFEST.json)、[配置独立比对收据](CONFIGURATION_VERIFICATION.json)、[笔记本独立核验](INDEPENDENT_LAPTOP.json)、[本机独立核验](INDEPENDENT_LOCAL.json)、[回传收据](TRANSFER_VERIFICATION.json)、[执行退出收据](EXECUTION_CLOSURE_LAPTOP.json)。
