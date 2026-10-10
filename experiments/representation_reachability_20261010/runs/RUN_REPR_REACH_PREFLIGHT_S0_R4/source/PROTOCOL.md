# 表示层可达性与局部 prototype residual

**研究线**：`neighbor_sensitive_p3`  
**状态**：执行前冻结；协议与四臂对照已定义；工程筛查尚未启动。  
**冻结日期**：2026-10-10  
**结论范围**：本协议只回答“局部系数残差是否有效，以及 prototype residual 是否在局部系数之外提供增益”。它不把一次小样本筛查升级为最终论文结论。

## 1. 问题与假设

研究问题是：在固定检测框、类别、候选身份和原生 prototype 后，局部图像信息能否把实例 mask 的系数与 prototype 表示推进到原生头未能到达的区域。

预注册假设：

1. `F_local_feature_coefficient_residual` 相对 `C_native_coefficient_capacity` 的增益，表示局部特征为系数预测提供了有效信息。
2. `R_local_proto_residual` 相对 `F` 仍有稳定增益，表示低秩 prototype residual 扩大了表示可达空间，而不只是增加系数头容量。
3. 所有增益必须在未参与训练的 COCO val2017 上复现，并在至少两个外部数据集上不出现系统性反向。

旧的局部原型诊断实验只作为先验和实现参考；不复用其旧缓存，也不把 h-only、同图错配或错图错配诊断臂并入本研究四臂。

## 2. 固定模型、软件与训练边界

- 模型：官方 `YOLO26m-seg`，权重 `yolo26m-seg.pt`。
- 权重 SHA256：`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`。
- Ultralytics：`8.4.100`；输入 `640`；`FP32`；`mask_ratio=1`；`overlap_mask=true`。
- 每个训练臂从同一官方权重独立初始化。A 不训练，直接作为官方冻结推理参考。
- one-to-many 与 one-to-one 两条分支都必须接入 C、F、R；只接入一条分支时该 Run 无效，不得补写为完整结果。
- 检测、分类、box 回归、候选排序和 NMS 路径冻结。C、F、R 只允许改变 mask 表示路径。
- 训练数据顺序、增强、优化器、学习率日程、epoch、梯度累积、AMP 设置和评估代码在四臂间相同，并写入 Run 的 `TRAINING_SETUP`。正式比较不使用 best checkpoint，主结果固定为最后一个 epoch。
- 禁止 GT box、GT mask、GT 类别或任何由 GT 生成的 ROI 进入推理预测器；GT 只用于训练损失和离线评估。

执行环境尚未选择。创建任何 Run 前，必须把解释器、GPU/CPU、数据路径、数据版本、依赖锁定和权重哈希写入 Run；本文件不假设租用服务器或本地笔记本。

## 3. 数据与划分

主数据集为官方 COCO `train2017`/`val2017`。正式训练集固定为 20,000 张：对 `train2017` image ID 计算

```text
SHA256("representation_reachability_train20k_seed0:" + str(image_id))
```

按哈希、再按 image ID 升序排序，取前 20,000 张。每个 Run 保存完整 image-ID 清单和哈希，禁止按结果重抽样。验证固定为完整 COCO `val2017` 5,000 张。

PigLife 与 Faro 只在 COCO 正式门槛通过后进入外部验证，但数据版本、路径和清单在 Run 启动前就要解析并记录；不得在看到 COCO 结果后选择性更换版本或子集。

工程筛查使用固定小切片验证形状、梯度和冻结边界，不用于科学结论：32 张 fit、8 张 val，固定 image-ID 清单，四臂同一清单。工程筛查通过后才允许创建 20k 正式 Run。

## 4. 四臂定义（唯一变量）

### A — `native`

官方 YOLO26m-seg 冻结推理。所有参数冻结，输出原生 prototype `P`、原生系数 `c`、原生候选框和分数。A 的作用是参考和候选身份基线，不产生训练梯度。

### C — `native_coefficient_capacity`

只解冻原生系数头 `model.23.cv4` 与 `model.23.one2one_cv4`，其余模型冻结，包括 prototype、检测、分类和 box 路径。训练损失、输入、候选和推理后处理与 F/R 相同。C 用来测“原生系数头容量/适配”本身，不能被解释为局部信息方法。

### F — `local_feature_coefficient_residual`

冻结官方 prototype 和原生系数头，在进入每条原生系数头的特征图上增加一个局部 ROI 残差头。对预测框在该特征图上的 ROI 做 ROIAlign 或等价面积平均，固定输出 `4x4`，展平后使用

```text
Linear(D*16, 256) -> SiLU -> Linear(256, 32)
```

其中 `D` 是该分支实际输入通道数，必须在 `TRAINING_SETUP` 中记录。该头输出 `Delta-c`，mask 使用 `P(c + Delta-c)`；`P`、`c` 和检测候选均来自冻结官方路径。one-to-many 与 one-to-one 各有一套独立残差头，结构和初始化规则相同。ROI 只使用预测框，训练时不注入 GT ROI。

### R — `local_proto_residual`

R 包含 F 的全部局部系数残差，并增加低秩 prototype residual。prototype 分支从原生 prototype 输出前的冻结特征张量接入一个 `1x1` 投影，产生 `K_r=4` 个残差 prototype 通道 `Q`；局部 ROI 头额外输出 4 维 `Delta-d`。最终 mask logit 固定为

```text
P(c + Delta-c) + Q Delta-d
```

原生 `P`、检测候选和原生系数路径保持冻结；只有局部残差头与 `Q` 投影可训练。`Delta-c` 与 `Delta-d` 的最后一层零初始化，确保训练起点精确退化到冻结参考；`Q` 投影采用预注册的 Kaiming uniform 初始化。R 的增益只有在超过 F 时才归因于 prototype 表示残差。

参数量不强行伪装成完全相等：C 是原生容量控制，F 是局部系数控制，R 是 F 加 prototype residual。四臂使用相同数据顺序、优化器预算、epoch、评估和最终 checkpoint 规则；每臂在 Run 中报告可训练参数量和冻结审计。

## 5. 工程筛查（G0）

工程筛查只检查“能否正确运行”，不报告科学增益，不创建正式结论。建议 Run 命名为：

```text
RUN_REPR_REACH_PREFLIGHT_S0
RUN_REPR_REACH_SMOKE_A_S0
RUN_REPR_REACH_SMOKE_C_S0
RUN_REPR_REACH_SMOKE_F_S0
RUN_REPR_REACH_SMOKE_R_S0
```

必须全部满足以下条件才算 G0 通过：

1. 四臂在固定 32/8 切片上前向、反向和保存 checkpoint 均无 NaN/Inf；C、F、R 的预期参数确实收到梯度，冻结参数梯度始终为空或为零。
2. A/C/F/R 的候选框、类别、排序、置信度和 NMS 结果逐项一致；只有 mask logit 路径可不同。
3. F/R 的 ROI 来自预测框，代码审计确认推理路径没有读取 GT；one-to-many 与 one-to-one 均有独立输出。
4. R 在 Delta 头零初始化时与冻结参考的 mask logit 逐元素一致到预注册数值容差；Q 分支和 shape 检查通过。
5. 反解码后的 mask、RLE、面积统计和 COCO evaluator 在 8 张 val 上可复现；固定随机种子、版本、清单和张量 shape 全部写入 Run。

任一项失败，停止所有正式训练，修复后重新创建新的 Smoke Run；不覆盖失败产物。

## 6. 正式 20k 训练

G0 通过后，四臂各创建 seed 0 和 seed 1 的独立正式 Run。A 只需保存两次相同配置的验证记录，C/F/R 按固定 20k 训练 8 个 epoch；若实际训练预算必须改变，四臂必须同步改变并在启动前冻结。每个 Run 保存：训练 image-ID 清单、环境快照、权重哈希、冻结审计、参数量、每 epoch 指标、最后 epoch checkpoint、完整 COCO val 预测和失败/部分产物。

不做基于 val 的扫参，不用 best checkpoint，不因单个 seed 结果改协议。seed 0/1 的结果先按预注册指标合并，再给出 seed 间离散度；seed 不足或 Run 失败时保持未知，不补造结果。

## 7. 评价指标与统计

主指标为 COCO instance Mask AP@[0.50:0.95]，单位为 AP points。次指标为 Mask AP50、AP75、APsmall，以及 Box AP 用于确认候选身份没有漂移。所有臂保存逐图结果，使用 1,000 次按 image 重采样 bootstrap 报告差值的 95% CI；不能把实例级重复当成图像级独立样本。

同时报告：成功实例的损伤率、空 mask/异常面积率、按目标尺寸分层的差值、one-to-many/one-to-one 分支差值，以及外部 PigLife/Faro 的同口径结果。原始表示诊断（系数可达性、residual 范数、ROI 覆盖率）只作机制证据，不替代 mask AP。

## 8. 预注册决策门槛

- **G1 主效应**：R 相对 A 的 COCO Mask AP 至少 `+0.3` AP points，且 image-bootstrap 95% CI 下界 `> 0`。否则停止把 R 作为主方法继续扩展，结论为未达到主线门槛。
- **G2 表示层增益**：R 相对 F 至少 `+0.1` AP points，且 CI 下界 `> 0`。G1 通过而 G2 不通过时，只保留“局部系数残差”解释，不声称 prototype residual 带来独立增益。
- **G3 安全性**：AP75 与 APsmall 均不得下降超过 `0.1` AP points；成功实例损伤率的 bootstrap 95% CI 上界不超过 `1%`。任一失败都不能按主方法推进。
- **G4 外部稳定性**：在 PigLife 与 Faro 上，R 相对 A 不得同时为负；至少一个外部数据集的机制相关子组需保持非负或改善。若外部数据不可用，结论范围限于 COCO。
- **臂间解释**：C 相对 A 的变化用于容量控制；F 相对 C 的变化用于局部系数信息；R 相对 F 的变化才用于 prototype 表示层结论。不得用 R-A 的单一差值替代这三个对照。

所有门槛均以最终预注册指标判断。工程 Smoke、单 seed、局部子集或仅诊断指标不能标记 G1–G4 通过。

## 9. 可复现性、停止与限制

本协议冻结后，任何修改必须创建新版本目录或新 study，不得静默改写本文件。失败 Run 和部分产物保留；结果未到达、已到达未读、已读未处理和远端状态未知要分别记录。未选择执行环境、未解析外部数据版本或未通过 G0 时，不创建正式训练 Run。

本研究只测试一个固定的低秩 residual 设计（`K_r=4`）和一个固定的局部 ROI 结构；阴性结果不能证明所有 prototype 表示方法都无效，阳性结果也不能证明全局最优。任何超出 COCO 20k、PigLife/Faro 和本协议四臂的扩展，都必须另立实验并重新冻结协议。
