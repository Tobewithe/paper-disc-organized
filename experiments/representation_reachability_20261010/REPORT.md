# G0 工程筛查记录

**结论**：G0 工程预检和 A/C/F/R 四臂 Smoke 均通过；允许进入 20k 正式训练准备，禁止把 Smoke 结果解释为 Mask AP 或科学结论。

## 通过证据

- 环境：本地笔记本 `28358lan`，CUDA `NVIDIA GeForce RTX 4060 Laptop GPU`，Torch `2.5.1`。
- 真实导入：Ultralytics `8.4.100`，来源为 `D:/coco_wire/vendor_8.4.100/ultralytics/__init__.py`。
- 权重：官方 `yolo26m-seg.pt`，SHA256 `16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。
- 数据：固定 32 张 fit 与 8 张 val；32 个 fit batch 和 8 个 val batch 均有限且可完成前向。
- 原生结构：`model.23.cv4`、`model.23.one2one_cv4`、`model.23.proto` 均存在；C 只训练 1,709,760 个 native coefficient 参数，冻结 25,402,312 个参数的状态在 optimizer step 后保持不变。
- 梯度与残差：native coefficient 最大梯度绝对值 `3.903897285461426`；F/R 残差模块梯度有限；F 零初始化基线一致；R rank 为 4；候选身份张量保持一致。

## 保留的失败 Run

`RUN_REPR_REACH_PREFLIGHT_S0` 到 `_R5` 的失败产物均已保留。失败分别来自包装器缺少可选文件、LetterBox 调用方式、可微前向路径、GPU batch OOM 和断言层级错误；没有一个失败 Run 产生科学指标。

## 四臂 Smoke

- A：`RUN_REPR_REACH_SMOKE_A_S0`，32/8 前向有限，0 个可训练参数。
- C：`RUN_REPR_REACH_SMOKE_C_S0`，1,709,760 个 native coefficient 参数收到有限非零梯度，冻结状态未变。
- F：`RUN_REPR_REACH_SMOKE_F_S0`，20,768 个残差参数梯度有限，零初始化基线和候选身份通过。
- R：`RUN_REPR_REACH_SMOKE_R_S0`，13,588 个 residual 参数梯度有限，零初始化基线和候选身份通过。

四臂均使用同一官方权重、同一 32/8 图像清单和真实导入的 Ultralytics 8.4.100。

## 范围限制

本阶段只验证官方模型加载、固定 smoke 图像前向、native coefficient 冻结边界、F/R 残差张量与梯度路径。尚未执行 COCO evaluator、Mask AP、外部数据或 20k 正式训练。

下一阶段只能创建独立的 `RUN_REPR_REACH_SMOKE_A_S0`、`_C_S0`、`_F_S0`、`_R_S0`，并沿用本协议的固定输入、候选身份和最终 epoch 规则。
