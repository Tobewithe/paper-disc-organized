# 全量猪群基线合同（5点目标的共同起点）

用户2026-10-23要求优化原型/训练并设定至少+5点目标。本合同仅冻结纯基线与真实缓存；方法训练另按TASK_PROTOCOL_V2.md冻结，不启动旧P3门控正式训练。渲染任务194/195继续沿用原始PROTOCOL.md，不改其冻结记录。

1704张原始train图完整使用，按既有13/4 family分为1272 train/432 dev；原426 test图不读取。训练120轮×106步=12720步，batch12；每轮将1272图完整洗牌恰好一遍，不以无限有放回抽样冒充完整epoch。初始化固定COCO yolo26m-seg.pt SHA16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5，Ultralytics8.4.100、单类pig、原生O2M/O2O/Mask/Semantic损失、原生decoder保持。

训练使用A6000 AMP bfloat16加速，模型参数/优化器FP32、评价FP32，无GradScaler；先2步工程核验非有限/梯度及可重载，再正式训练。AdamW peak lr .0003、wd .0001，300步线性warmup(.1→1)、cosine到.02，clip10；seed0，不早停、不追加预算。E2ELoss以总step作进度，逐步update。

RGB固定PIL bilinear640letterbox/JPEG95，Mask由原始COCO原分辨率解码后nearest映射同画布；训练使用同一RGB亮度/对比与水平翻转，各组同样策略。实例GT保留原生visible mask/RLE及归一化box，训练mask160。暂不加入未核验的仿射/mosaic，避免更改标注语义。

每1060步及最后评价同432张开发图；conf .001、max_det300、process_mask原生upsample/crop/logit0、COCOmaxDets1/10/100，segm评价去bbox。最高dev Mask AP选优、平分取早，保留last与恢复状态。不把全量新432图分数与旧54图直接作涨幅。B*是随后原型/所有方法臂共同起点，方法与相同数据/优化步数continued比较+5pp目标。

单GPU最多8小时，6CPU/32GiB；原始输入/缓存/源码/权重与训练进度SHA真实记录。基线性能不是原型贡献，结案仍按问题—机制(不适用)—方法—结果—研究决策交付。
