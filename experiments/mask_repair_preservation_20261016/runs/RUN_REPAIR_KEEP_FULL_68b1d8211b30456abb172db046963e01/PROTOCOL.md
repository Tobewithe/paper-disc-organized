# 同检测系数修复—正确像素保护：有限机制协议 V1

2026-10-16。替代本 Study 的[未执行设计草案](PROTOCOL_DRAFT.md)，不改任何旧 Run。正式冻结时间及每个文件的实际字节 SHA 以 `FREEZE_MANIFEST.json` 为准；只有输入预检与合成工程测试通过才生成该清单。`CONFIG.json` 是完整机器参数，不接受运行时科学参数覆盖。

**授权与范围：** 用户在当前状态汇报后要求“继续吧”，继续当前唯一下一步：补齐并冻结本项有限系数机制检验，具备条件后在本会话已授权的笔记本 `YANGLEI / ssh 28358lan` 执行一次面板外 smoke 和一次固定256条评价。只优化每条实例的32维临时系数变量；网络训练步数、模型/adapter forward 均为0，不更新已有 checkpoint，不自动进入三臂训练、S2/S3/S4、扫参数或提交推送。原讨论授权仅设计的边界由这次继续指令扩展到本项核验。其它研究任务、历史协议与失败均保留。

## 1. 问题—机制假说—针对性干预

问题：原 S1 256条中122条已有同检测 fixed-P Mask75见证；机会核对中100条未由adapter修到Mask75、10条修复被门控撤回、12条保留。特定BCE异常样本的像素记账又显示，降损可集中在未翻转像素，而前景丢失造成二值IoU退化。AMT-v1取得原生收益但相对原soft-IoU+面积门控增量为负，不能据此推论“所有目标都无效”或“已证明是样本太少”。证据和 Run 见[草案第1节](PROTOCOL_DRAFT.md#1-主判断与已有证据)。

**H：**显式区分“推动原错误像素越过阈值”和“维持原正确像素的已有间隔”，可在共享32维系数的联动中取得更好的修复—保护折中。**竞争解释：**保护表现只来自改动幅度降低；或者该目标干预没有额外效用，真正未兑现环节仍在条件方向预测。H是待检验目标机制，不是已经定位的唯一根因，也不是不可避免的32维容量冲突证明。

针对性干预：C为同网格BCE；E只集中修错；EK在完全相同E上加入保护；E-equal沿E方向一次缩放到EK的实际logit改动幅度。EK−E是保护项移除对照，EK−E-equal排查幅度解释；EK−C检验该复合目标是否值得进入真实学习。每臂同32变量、同c0初始化、同数据/几何/正则/求解预算。新增网络参数为0。有限oracle阳性不等于网络能学会或部署安全。

## 2. 样本、参照、身份与冻结输入

- 正式面板：原 `RUN_S1_ORACLE_f35a1399ac244f74aeebf94e4e843949/TARGET_PANEL.json` 的全部256条、243图；SHA `170676d3017e4be7cd213ff1094ca28065727807c915b7602ecdaec48238cf31`。不只选择122或100，不删失败/数值未收敛项，不换检测。
- 原生 N：`RUN_NATIVE_FULL_7609b70db27a4989b8da814e51cbd170` 的同一最终 Box75 检测；GT为原生COCO segm75任务失败且该检测自己的原图 MaskIoU<.75。保留预测namespace、全局行号、图/类/GT/raw身份、框、分数、排序、全部N行系数矩阵形状。
- 指定已有方法 R_G：原soft-IoU checkpoint加固定原生面积≥9216门控，来源全量 `RUN_S3_FULL_670fffd010664a648e096572fc7c476d`，复用已独立核验的 `RUN_OPPORTUNITY_FULL_18ab59a063b0489d81f541d45a907bfd/PAIRED_MASKS.jsonl` 中同256条掩码。
- 历史有限参照 R_S1：该面板原S1 proto-BCE oracle；仅历史见证。新C在输入尺度而历史S1在proto尺度，不能把两者的差归因为保护项。
- 每次报告 EK−N、EK−C，并单列 EK−R_G、EK−R_S1。后两项是同检测IoU描述性比较；oracle使用GT，不能宣称优于无GT部署方法或把增量称作AP。固定256条没有原成功实例总体，因此不作22957个成功GT的损伤保证。
- 四个 smoke：在原生账本中选正式243图之外、满足同一Box75/Mask75失败定义的记录，按 `(image_id,annotation_id)` 升序选前四个不同图。不查看任何新求解结果来选样。具体ID随 `PREPARATION.json` 在求解前冻结；不并入正式分母。
- `INPUT_EXPECTATIONS.json` 固定原生权重、完整COCO GT、原生输出/账本/trace清单、原S1面板及历史参照字节SHA。预检另核247张图及cache的SHA、实际geometry；运行再次核输入和cache。GT仅离线目标/测量，未调用任何预测网络。

## 3. 原生解码、ROI和浮点合同

核准源码为 Ultralytics8.4.100：`process_mask_native` 用**完整N×32** FP32 GEMM形成proto logits，按 `max(1,32000000//(H*W))` 分块 `scale_masks` 到 `native_decode_hw`，严格`>0`转uint8，再用输入坐标预测框裁剪。随后 `SegmentationValidator.scale_preds` 用缓存ratio_pad对**二值**掩码 `scale_masks` 回原图并`.byte()`；禁止换成最终0.5阈值。实际版本/源码和运行数值开关与原生合同逐字段相等。

所有新臂的首次阈值网格都是该实际输入网格（缓存断言`native_decode_hw=input_hw=4*proto_hw`）。ROI严格按像素坐标 `x>=x1,x<x2,y>=y1,y<y2`。原始COCO GT nearest重采样到缓存真实resized尺寸，再按真实整数left/top/right/bottom补0；奇数padding不擅自改成对称。原图主IoU总是完整GT RLE，包括框外GT。

**为避免伪双精度，明确一个共同代理：**固定版`ops.scale_masks`内部调用`.float()`，直接把FP64系数送入它不能宣称全FP64求解。因此仅优化端使用P通道的等价实数双线性插值（FP64、align_corners=False）得到B，ROI内目标为 `z(delta)=z0_native_FP32.double()+B*delta`。原生c0处由真实完整形状GEMM的logit锚定，E/K分组不会被代理舍入改变。该代理对所有臂完全一致，属于本次事前明确的数值实现；不是改变实际decoder。

返回系数统一 `(c0.double()+delta).float()` 放回完整N行；实际输入二值掩码与原图RLE均经原生FP32算子得到。每个像素核代理和实际logit差 `<=5e-4+5e-5*abs(actual)`；记录最大误差、越限像素和阈值翻转差。代理的阈值不用于最终评分。任何越限、非有限量、等幅失配均不能支持机制阳性；非有限量或身份/重放失配为工程失败。

每图zero-delta经原生bridge精确重放全部输出RLE；自定义分阶段捕获还须与bridge逐像素相等。所有变体保留非目标输出的首次二值与原图mask逐像素相等，框/类/score/ID始终来自未改动原生行。

## 4. 三目标、分组及等幅对照

ROI内GT为y，s=2y−1。以**实际c0首次阈值**一次固定 E=`(z0>0)!=y` 和 K为其补集，迭代不重分组。各非空组各自取均值，空组项0；空ROI实例不删，其所有系数更新为0，列出数量。

| 臂 | 唯一操作 |
|---|---|
| C | `mean_ROI(BCEWithLogits(z,y)) + .003*||delta||²/2` |
| E | `mean_E(relu(1−s*z)²) + .003*||delta||²/2` |
| EK | E加 `mean_K(relu(min(1,s*z0)−s*z)²)`，权重1 |
| E-equal | 已解E方向乘 `RMS_ROI(actual_z_EK−z0)/RMS_ROI(actual_z_E−z0)` 一次，再原生解码 |

K间隔在c0时保护损失为0，不奖励已满足间隔像素继续增信。E与C是复合目标对比，不声称单独识别hinge/组归约/像素选择中的某一个因素。`softplus(-s*z)`等价BCE，不能换写法当新机制。

等幅比值来自实际FP32logit，非GT成绩；允许放大>1，不限幅，不做二次搜索/矫正。两者RMS都精确为0，取alpha=0，记both_zero；E为0而EK非0，记undefined_zero_direction，仍保留该行，整体数值不可裁决。重放后RMS误差须`<=1e-4+1e-4*RMS_EK`，不达标不修改容差补跑。

## 5. 求解预算、工程控制与数值裁决

CPU单线程、FP64、delta=0、Torch LBFGS/strong_wolfe：lr1，120迭代，150次目标/梯度closure硬上限，history20，tolerance_grad1e−9，tolerance_change1e−12；最终梯度无穷范数≤1e−6记收敛。达到closure硬上限时保留已经计算过的最低**目标值**迭代点，标明cap，不按GT IoU选解；末次梯度仅作收敛检查、不再更新。三臂完全相同。

数学工程控制在任何真实样本求解前完成：三目标有限差分梯度、C/E/EK各自非零已知最优解、固定E/K与保护项初始0、空组/空ROI、等幅零/放大规则、线性插值与奇数padding、真实byte解码区别、TP/TN/FP/FN独立计数、图宏归约和分母缺失拒绝。合成测试不算真实模型或科学Gate通过。

最多smoke4条×3解、≤600秒；正式256×3=768解、≤7200秒；每Run输出≤2GiB、RSS<6GiB、空闲RAM≥768MiB、磁盘余量≥2GiB。启动前空闲RAM≥3GiB且磁盘≥4GiB。每closure、每图核预算；不因结果追加迭代、换seed或补样。smoke全部工程/数值/独立核验通过，且 `1.2×smoke逐图处理耗时×256/4+120秒≤7200秒`，才执行完整面板。预算不足只结案工程限制，不能自动扩预算。

所有256条进入图宏统计，且主量、保留和保护判据都依赖完整面板；任一用于正式比较的求解未收敛或数值代理/等幅失败，本次整体记 `INCONCLUSIVE_NUMERICAL`。未收敛数据保留，不作不可达证明，不从有效分母剔除。此规则补齐草案中“是否影响裁决”的歧义，事前一次固定。

## 6. 统计、门槛与三个出口

每个指标先同图实例均值，再243图等权。按image_id升序形成向量；NumPy default_rng(1729)生成2000组243图有放回索引，同一索引用于全部比较，双侧2.5/97.5百分位，quantile method=linear。面板已用于方法开发，区间仅描述开发证据，不具有新的盲测确认含义。

必须同时满足：

1. EK−C原图平均IoU≥.005，配对95%下界>0。
2. 首次阈值网格 `TP→FN/max(1,native_TP)` 以同样图宏归约；EK较E-equal均值相对下降≥25%，**绝对下降量**的95%下界>0；EK−E-equal原图IoU≥−.001。native_TP=0实例计0并列数；E-equal平均损失为0，不能声称相对下降通过。
3. C达到原图Mask75的实例中，EK保留≥95%，并列每个新增/丢失ID。C没有成功见证不视作保留通过。
4. 全部样本、实际原生重放、输入/身份、数值、源码、独立RLE/统计核验有效。

- **阳性 `FINITE_MECHANISM_SUPPORTED`：**四项全过，支持这个有限系数目标干预具有保护相关增益。唯一下一步为另冻同架构/同预算C/E/EK训练消融，检验能否学会；本轮不启动。
- **阴性 `CONFIGURATION_REJECTED`：**工程数值有效，但任何点值护栏不达标。关闭这个margin/保护目标配置，停止同类loss的串行尝试。优先聚焦已知有益方向的条件预测；先形成有自己证据支持的具体输入/读出命题，不自动恢复181维二分类、扩量或改门控。
- **信息不足：**点值均合格但要求的区间未过记`INSUFFICIENT_PRECISION`；数值未决记`INCONCLUSIVE_NUMERICAL`；身份或执行错误为工程失败。均停止本Run并保留，不能偷偷放宽Gate或追加样本/预算。可验证的工程问题另行登记修复，不能把同协议已看到的结果包装成新盲测。

这些是机制探索投资门槛，不追溯修改任何旧S1/S2/S3。首次阈值保护属于失败实例内部正确像素；不替代原成功GT损伤，不计算新的COCO AP。

## 7. 配置、运行登记、独立验证与交付

沿用原生8.4.100官方权重及实际640/batch16/rect缓存，不重新分batch、不再forward。原生end2end、conf=.001、iou=.7、max_det300、agnostic_nms=False、mask_ratio4、overlap=True、retina=False、FP32与TF32等完整实际环境按源Run逐项核验；官方权重训练元数据mask_ratio1不进入本次评价。官网52.5/44.1 parity未过与native账本有效双状态保留。

每次实际执行才登记独立Run、原命令、主机/PID、父Run、protocol/code/input哈希及源码快照；原子阶段锁防重复。输入预检和源码冻结先完成，再smoke，再一次正式Run。同阶段已存在Run即拒绝自动重跑；不终止其它会话任务。

原始逐实例结果保存c0、三解及等幅delta/FP32系数、收敛/实际RMS/代理误差、首次阈值GT/原生/新掩码RLE、完整原图GT/各参照RLE及像素迁移计数。`verify.py`不用优化器代码，以独立压缩RLE解码、像素计数和图宏bootstrap复算统计及Gate；笔记本完成后回传全Run（含失败、源码快照）并核SHA，本机执行同一冻结verify脚本再核统计。此独立检查不冒称在本机重跑GPU P解码。

产物：PREPARATION/FREEZE、Run的EVAL_CONTRACT、PAIR_RESULTS、STATISTICS、CONFIG_AUDIT、SUMMARY、DECISION、INDEPENDENT_LAPTOP、本机INDEPENDENT_LOCAL、回传清单、REPORT及配置复核；更新Study和总控，运行 `python project/check_active_plan.py`。结案按“问题—机制假说—针对性方法—实验反馈—唯一下一步”，同时说清原生收益、匹配C增量及与已有方法的比较范围。
