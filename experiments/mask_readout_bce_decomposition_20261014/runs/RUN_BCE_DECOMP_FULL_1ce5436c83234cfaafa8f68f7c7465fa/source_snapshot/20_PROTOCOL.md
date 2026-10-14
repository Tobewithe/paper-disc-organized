# 冻结 checkpoint 的 proto ROI 二值质量一致性审计

2026-10-14，用户授权笔记本 YANGLEI；只使用既有 B checkpoint、原 fit2000 缓存和上一诊断的固定监督配对，新增训练步数为 0。生效协议是各 Run 内启动前保存的本文件与 EVAL_CONTRACT/SHA；不更改旧训练、旧诊断或 S2 阴性判定。

## 命题、总体与唯一主量

量化已知 3950 个“GT BCE 下降而原图二值 IoU 下降”的监督实例中，有多少在同一 proto 网格、同一训练 ROI 的二值 IoU 已经下降。逐对补齐全部 14268 个有效监督配对，列出 proto IoU 下降/持平/改善的互斥计数；这定位首次观测到质量反转的区间，不把两级指标差异直接称为 decoder bug。

来源为 RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926 的 TRAINING_PAIR_EFFECTS、LOSS_BY_IMAGE、ROW_MAPPING、ZERO_DELTA_AND_JOIN_CHECKS、CONFIG_AUDIT 和完整预测文件。输入 SHA 在 INPUT_EXPECTATIONS.json 与每个 Run 合同固定。已有原图 IoU 与零残差 218209 行精确 RLE 重放作为经哈希复核的端点证据复用，本次不重算 COCO AP 或改变输出。

## 测量定义

- 固定原 B last.pt（SHA 201ab6a7372c167e772217904eac8f4ba9946c26e5862c2882e3c01bdcd58e5e）、原 P/c0/F、模型 eval、FP32、无 AMP、无梯度、无优化器；计算 c0 和 c0+Δc 的全输出系数矩阵后再选监督行，避免矩阵批形状改变数值。
- 使用原 adapter_model.loss_grids 产生 logits 和 ROI，原 adapter_data.projected_targets 投影同一 GT。ROI 是预测框映射到 proto 网格的左上闭、右下开整像素集合；预测二值条件严格为 logit>0，GT 必须为 0/1。TP/FP/FN/TN 只计 ROI 内。
- 二值 IoU=TP/(TP+FP+FN)。并集为 0 时定义 IoU=1，并逐臂标记、单列数量；空 ROI 依原训练规则排除且逐图记账。不得丢弃持平实例或微小目标。
- BCE 方向容差沿用 1e-6；proto 和原图 IoU 的方向容差均为 1e-12。逐对保存整数混淆计数、ROI/GT 像素数、两臂 IoU 与三阶段方向。训练 join 按保存的 class+box50+score 一对一关联逐图重构并相等核验，不改用 bbox75/segm75 匹配。
- 重新计算原 GT BCE、保持项与总 loss，要求每个值与上一诊断的绝对误差≤1e-7；复核全部有效配对主键、图像与原输出次序相等。超限即工程失败，不继续科研判定。

## 判定与后继

主量是上述 3950 个固定异常配对中 proto IoU 已下降的比例，属于既有 fit 总体全查，不作独立泛化置信声明。全部覆盖且配置审计通过才判定。

若该比例严格大于 50%，采用“主要反转已在同 ROI 的 BCE→二值质量之间出现”，唯一后继为零训练分解这些实例中前景/背景 BCE 贡献及 TP/FP/FN 变化，确定 loss 下降如何伴随二值错误增加。否则拒绝“多数反转已在此级出现”，唯一后继为零训练核查剩余实例从 proto 到原生上采样、裁切及原图评价范围的逐级转换。50% 是预先冻结的多数定位/工作优先级门槛，不是新方法效用 Gate；持平与改善分别报告，后一级结果不用于倒推前一级无影响。

推翻条件：任何配对身份、GT 投影、BCE 重算、冻结输入或源码核验失败；触发后撤销本次定位，唯一后备动作是修复测量实现并另建 Run 重做零训练审计。S2/S3/S4 不因本诊断放行。

## 执行与结案

先通过计数、阈值边界、空并集、三方向与判定门槛的行为测试，再用原 fit_ids 前 8 图做独立 smoke，正式 Run 使用全部原 2000 图；smoke 和正式 Run 的源码、输入与环境必须一致。预算 smoke≤900秒、正式≤1800秒、产物≤1GiB；逐图处理，单线程 CPU、RTX4060 Laptop CUDA。保持原缓存 batch16/rect=True/imgsz640 的实际几何，mask_ratio=4 等完整基线合同及原训练 loss 不变，不新建基础模型推理批次。

完成后核对所有缓存及输入前后 SHA、运行时/包/原始源码、checkpoint 内存状态、零梯度/零新增步数，保留 run.json、原日志、逐对计数和 CONFIG_AUDIT。回传 manifest/transfer，并在本地从整数计数与原表独立复算 IoU、全部分类与唯一判定；报告后等待用户安排后继，不自动开始另一项实验。
