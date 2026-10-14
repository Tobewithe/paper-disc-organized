# YOLO26m-seg 基线配置合同

采用 v8.4.100 官方原生 `YOLO(...).val()`、同 SHA 的官方权重和原始 COCO val2017，以原生 `COCOeval_faster` 的输出作为发表指标复现依据。机器可读值见 [baseline_profile.json](baseline_profile.json)；Run 保存全量解析参数和实际运行回执，不只保存本表。

来源：[固定版本官方表](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/macros/yolo-seg-perf.md)、[官方复现说明](https://github.com/ultralytics/ultralytics/blob/v8.4.100/docs/en/tasks/segment.md)、固定版本 `cfg/default.yaml`、`engine/model.py`、`engine/validator.py`、`models/yolo/{detect,segment}/val.py`。官方发表 Box/Mask AP 为 52.5/44.1；未公开完整软件栈和硬件数值设置，保留这一来源限制。

| 项目 | 来源差异及统一决定 |
|---|---|
| 权重与版本 | 权重 SHA 以 profile 为准；权重训练元数据版本 8.3.222，执行源码固定 8.4.100，二者不混称同一版本 |
| 参数继承 | 加载 `.pt` 后仅保留 imgsz/data/task/single_cls 等少数覆盖；不自动继承全部 train_args。Run 同存原始 train_args、框架默认、请求参数与实际参数 |
| 批量与几何 | 默认 batch=16；`model.val()` 覆盖 rect=True（全局默认 False）。采用原生矩形批次、stride 对齐、val pad=0.5，逐图保存实际尺寸/ratio_pad/顺序；640 不保证每张实际输入均为 640×640 |
| 分支与选择 | end2end=True、agnostic_nms=False、max_det=300、conf=0.001（严格大于）。iou=0.7 保留默认，但 E2E 不执行常规 NMS；导出参数 nms 不负责选择 O2O |
| 数值与融合 | 原生 PyTorch CUDA FP32、fuse=True、eval 模式；quantize=None。固定本机原生数值状态：matmul highest、CUDA matmul TF32=False、cuDNN TF32=True、benchmark=False；实际确定性算法开关 False。配置中 deterministic=True 不代表 standalone val 已调用训练的 init_seeds |
| mask_ratio / overlap_mask | 权重训练记录 1/True；本版本 val 默认 4/True。本次采用 val 默认。二者影响内部 GT 栅格与内部指标，原生导出的预测及外部 COCO 指标必须由原始标注核验；不能根据内部指标与外部指标差异调 mask_ratio 追分 |
| mask 解码 | save_json=True 选择原生 process_mask_native；先阈值/裁剪，再将二值 mask 按原生逻辑缩回原图。retina_masks=False 是 predict 参数，不能当 val decoder 开关 |
| 评价后端 | 原生 faster-coco-eval=1.8.0；独立 pycocotools=2.0.11。完整 bbox+segmentation JSON 原样输入两者，检测 area 取 bbox；bbox/segm 各自匹配 |
| 评价预算 | maxDets=[1,10,100]、useCats=1、IoU=.50:.05:.95、crowd/ignore 原样；80→91 类别映射；每任务完整 12 项 AP/AR，单位为百分点评分 |
| 执行环境与差异 | 用户指定笔记本 YANGLEI：Python 3.12.3、Torch 2.5.1/CUDA 12.4、RTX 4060 Laptop；独立目录提供 Ultralytics 8.4.100，不修改旧环境 8.4.27。workers=0、plots=False、verbose=False，关闭仅 I/O/绘图负担；保留原生批次顺序、模型和评价路径。所有 5000 图、原始标注 SHA 和图像文件 SHA 可核验 |
| 非推理参数 | 增强/优化器/epochs/损失/AMP 等训练设置不因出现在 args 中就在独立 val 生效；augment=False、无 TTA、无 compile、无类过滤、无多尺度推理 |
| 新方法训练 | 后续单独冻结训练合同；不将训练默认 mask_ratio=4 混入官方权重训练记录，也不把历史官方 optimizer/batch/epochs 自动复制成 adapter 配方。本轮不启动训练 |

观察器只捕获最终输出→P/c0/F 的身份与实际原生 evaluator 结果，不改官方排序、预测、解码或匹配。`model.val()` 返回的内部 SegMetrics 与原生 COCO 评估结果分别保存；原生评估失败必须报错，禁止静默回退内部 AP。

权重中实际保存的训练记录与 8.4.100 默认值有以下差异。来源为笔记本真实 smoke 的 [CONFIG_SOURCES.json](../../../experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_SMOKE_a6dc7e4d82d847579eaf2c4f94416d05/trace/CONFIG_SOURCES.json)，它同时保留全部字段，不能把下表当作完整可重训配方。

| 训练参数 | 官方权重保存的值 | 8.4.100 默认值 |
|---|---|---|
| epochs / batch / optimizer | 80 / 128 / MuSGD | 100 / 16 / auto |
| lr0 / lrf / momentum / weight_decay | .00038 / .88219 / .94751 / .00027 | .01 / .01 / .937 / .0005 |
| warmup_epochs / warmup_momentum / warmup_bias_lr | .98745 / .54064 / .05684 | 3 / .8 / .1 |
| box / cls / dfl 损失权重 | 9.83241 / .64896 / .95824 | 7.5 / .5 / 1.5 |
| mask_ratio / overlap_mask | 1 / True | 4 / True |
| hsv_h / hsv_s / hsv_v | .01315 / .35348 / .19383 | .015 / .7 / .4 |
| degrees / translate / scale / shear / perspective | .00012 / .27484 / .95 / .00136 / .00074 | 0 / .1 / .5 / 0 / 0 |
| flipud / fliplr | .00653 / .30393 | 0 / .5 |
| mosaic / mixup / cutmix / copy_paste | .99182 / .42713 / .00082 / .30413 | 1 / 0 / 0 / 0 |
| 旧版本额外训练键 | semseg_loss=True、o2m=.70518、muon_w=.4355、sgd_w=.47908、cls_w=3.48357、stride_ratio=1、topk=5、hungarian=False | 本版本 default.yaml 无同名键，不能盲目回灌 |
| 旧数值/导出键 | half=False、int8=False | 本版本改由 quantize 等入口控制，实际 FP32 另由模型回执断言 |

这些差异解释了为什么“采用默认配置训练”不等于“复现官方权重训练”。当前论文统一使用**同一官方冻结权重 + 本文件的原生推理/评价合同 + 单独预注册的新增模块训练合同**；基础模型不重新训练。后续训练合同必须明确增量参数、初始化、标签与损失、优化器、有效 batch/步数、增强、AMP、seed 和 checkpoint 选择，基线与方法共享推理/评价合同。若另行声称复现官方基础模型训练，旧键含义及完整训练代码仍是待补输入，本次权重元数据不能证明该训练配方已复现。

验收沿用 S0 冻结协议：官方两项 AP 四舍五入到一位小数分别为 52.5/44.1；同一 JSON 的双后端 24 项指标差均 ≤0.02 点；身份/逐 GT 不变量与 zero-delta 掩码逐像素一致。烟雾 ≤16 图，随后只运行一次 5000 图前向，差异诊断使用保存的输出。通过前不登记“官方已复现”；两种科学状态分别记录。

本配置首次完整执行为 [RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170](../../../experiments/official_gt_failure_rebaseline_20261013/runs/RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170/SUMMARY.json)：原生 Box/Mask AP 52.3514301541/44.0141748477，官方一位小数复现未通过；同一预测双后端24指标完全一致、全量零残差重放与 GT 账本有效性通过。当前身份为 `native_config_scoped`，不能写成“官方发布条件已完全复现”。完整参数对照见 [CONFIG_PARAMETER_AUDIT](../../../experiments/official_gt_failure_rebaseline_20261013/runs/RUN_HISTORY_RECONCILIATION_ea22c6c2ae3c49eda6899b1ea98f40f4/CONFIG_PARAMETER_AUDIT.json)。
