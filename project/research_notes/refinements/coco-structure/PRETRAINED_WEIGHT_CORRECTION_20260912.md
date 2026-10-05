# 官方预训练权重核查：mask_ratio 已为1

2026-09-12，用户追问“下载的权重不就是最强的吗”后，直接读取项目实际下载的官方 `yolo26m-seg.pt`。文件SHA256与原下载回执及诊断协议完全一致：`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。

checkpoint内保存的train_args：mask_ratio=1、overlap_mask=True、imgsz=640、epochs=80、batch=128、optimizer=MuSGD、lr0=0.00038；内部初始model名yolo26m-objv1-seg.pt。checkpoint date=2026-01-10T06:25:02.276044，version=8.3.222，epoch=-1表示发布文件剥离后的字段，不代表只训练零轮。该元数据不是独立复现完整历史，但足以纠正先前把当前模型当作默认mask_ratio=4训练模型的无根据前提。

2026-09-14 进一步从 Ultralytics 官方 assets v8.4.0 下载并读取全系列发布权重。五个 checkpoint 的 `train_args` 一致记录 `mask_ratio=1`、`overlap_mask=True`、`imgsz=640`：n（epochs=245）、s（50）、m（80）、l（60）、x（40），权重内版本均为 8.3.222。由此可确认 `mask_ratio=1` 不是 m 型号的偶然特例。官方文档中的通用 `default.yaml` 仍为 `mask_ratio=4`；通用默认配置与发布权重的历史训练参数不是同一个事实来源。

证据存档：`experiments/coco_clean_20260911/diagnostics/native_label_source_20260912/WEIGHT_TRAIN_ARGS_WITNESS.json`。其中train_metrics是权重内历史字段，不是本轮新评价。官方v8.4.143文档性能表：YOLO26m/l/x-seg COCO Mask AP分别44.1/45.5/47.0；因此m是官方中型强基线，并非家族最高精度或所有实例分割方法中的最高水平。源文档归档upstream_seg_perf_v8.4.143.md，不能将该表用作本项目当前实测成绩。

需要明确更正：

1. S017—S020用冻结官方预训练权重做临时系数诊断；clean-run的后续微调配置mask_ratio=4是另一回事。不能将该后续配置倒推成官方权重的原始训练配置。
2. S019/S020低网格与高网格oracle比较只说明给定P和标签时不同优化目标的achieved解有差别；不能证明官方模型失败源自从未获得高分辨率监督。官方权重记录已经使用mask_ratio=1，这种历史解释缺乏依据，应停止使用。
3. 正式4 vs1若执行，只能解释从这个已有ratio1预训练权重出发的不同微调策略表现；不能包装为首次给模型加入高分辨率监督。当前用户要求方法创新，未启动该训练。
4. 仍保留的证据是：在已经过官方训练的冻结表示上，GT辅助逐实例系数优化可以改进选定失败目标。但它释放共享预测函数约束、直接访问测试GT，不证明系数相似、头容量不足、训练错误或可部署新方法。后续研究必须定位高分辨率预训练之后仍存在的具体读出/优化/选择机制，不能继续以原始低分辨率监督缺失为主线。

此项更正属于执行者先前未查权重元数据造成的推断缺口，不归责用户。各既有数值保留其真实诊断配方，不篡改为新实验。
