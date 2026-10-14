# 冻结 checkpoint 的 fit2000 诊断结案

采用“训练目标下降，但训练内修复与保持均失败”的定位；继续关闭当前 B 配置，不进入 S3、扩量或追加 epoch。全部2000图的三项诊断门槛均失败，故不能将当前失败只归因于训练到开发集的泛化落差。具体分歧发生于监督指标还是后续解码/评价范围，尚需下一项最小测量定位。

2026-10-14，执行端 YANGLEI 笔记本。正式 [Run](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/run.json) 为 `RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926`，701.328秒完成，新增训练步数 **0**。[冻结协议](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/PROTOCOL.md)、[执行合同](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/EVAL_CONTRACT.json)、[完整摘要](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/SUMMARY.json)和[裁决](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/DECISION.json)均保留。

## 结果与门槛

| 项目 | 原 fit2000，本次完整测量 | 原开发集，复用 S2 结果 | 训练内诊断门槛 |
|---|---|---|---|
| 目标 T 的 image-macro ΔMaskIoU | **−0.6315 pp**；95%区间 [−1.1679, −0.1057] pp | −0.9956 pp；[−2.0109,+0.0114] pp | ≥+0.5 pp，且区间下界>0：失败 |
| 全部原生 bbox75 固定配对普通GT的 image-macro ΔIoU | **−0.4622 pp**；[−0.5843,−0.3399] pp | −0.7561 pp；[−1.0053,−0.5215] pp | ≥−0.1 pp：失败 |
| 原成功GT任务损伤 | **591/9748 = 6.0628%**，完整有限总体的精确比例 | 63/1024=6.1523%，单侧95%上界7.5010% | 本次精确损伤≤1%：失败 |

训练基线含15118个普通GT、180个忽略GT；T=2669，覆盖1040图。总体IoU有12097个固定配对GT，覆盖1978图。原成功S=9748全部评价，没有抽样或换分母。bbox与segm各自匹配，损伤依据完整segm75任务标签，不由单条检测IoU阈值代替。

两个数据总体分别相对各自保存的原生输出计算。训练目标均值相对开发集高0.3641 pp、总体高0.2939 pp，是描述性的跨总体差；两者仍为负，训练内三项门槛并未成立。图像bootstrap固定2000次、seed1729、percentile/linear；训练数据区间不能作为未见数据证据。开发集来源为 [原S2报告](../mask_readout_mechanism_v2_20261012/S2_REPORT_20261013.md)，未重跑、重抽或重新选择 checkpoint。

## 固定模型的监督损失与同配对证据

| 原训练目标组成 | 零残差 c0 | B 最终 checkpoint | 变化 |
|---|---:|---:|---:|
| GT BCE | 0.32068555 | 0.26094683 | −0.05973872 |
| 基线soft保持 BCE | 0.30186387 | 0.33310923 | +0.03124536 |
| 合成目标：GT + 0.1×保持 | 0.35087194 | 0.29425775 | −0.05661419，降低16.1353% |

这是一对固定状态在全部原训练图片上的测量，不是将训练中的epoch在线均值相减。原Box50监督join、预测框ROI和两个独立归约分母未改变。14张图没有合法GT监督，456条输出ROI为空，均按原规则记账，图片没有删除。

14268个非空ROI监督配对中，9958个的GT BCE下降；其中**3950个同时原图MaskIoU下降，占BCE下降配对的39.6666%**。这是同一GT、同一最终检测的量，说明观察到的背离不只存在于两个总体均值之间；它仍不单独证明某个具体像素变换或损失权重是因果根源。详细表：[逐图loss](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/LOSS_BY_IMAGE.jsonl)、[训练配对的BCE与原图IoU](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/TRAINING_PAIR_EFFECTS.jsonl)。Box50训练配对与Box75评价配对分别保存身份。

## 配置与真实执行核验

- 输入：原 B `RUN_S2_TRAIN_B_6c7bc7e8c0dc48509a733d11e60f0ee2/last.pt`，SHA `201ab6a7372c167e772217904eac8f4ba9946c26e5862c2882e3c01bdcd58e5e`；原2000图缓存 `RUN_S2_CACHE_FULL_61ce0f1ca0e6479ca6704fb2fcceabec`，manifest SHA `fcae975a0fec3b03d0911785af2f9067b8bbd9ad4e6713aef7ba14911c96329d`。图清单、GT子集和原训练join全部核验，未更换图片。
- 环境：原离线Python3.12.3、Ultralytics8.4.100、Torch2.5.1、CUDA12.4、pycocotools2.0.11。运行环境与原训练合同相等，原生源码pins、模型架构、loss、GT投影、decoder和原S2账本实现均核对原Run源码哈希。
- 推理：FP32、无AMP；原生导出batch16/rect=True/imgsz640的每图真实几何直接保留。逐图读缓存没有重新分组或改变padding。原`mask_ratio=4`、`overlap_mask=True`等配置保持原记录，监督用原先已声明的原始GT投影。
- **218209条最终输出全部通过零残差RLE精确重放**。B只改变掩码，Box/Class/score/顺序、bbox75检测身份不变；GT和ignore域完整，bbox/segm评价独立，完整JSON保留bbox面积口径。
- B参数全部冻结、eval及inference_mode；未创建优化器、未反传、未保存新checkpoint。在内存的模型state、基础权重文件、checkpoint、缓存、原预测、GT和全部冻结输入前后SHA相同。原checkpoint仍记录原1500步，本诊断新增步数0。
- 本地和笔记本各4项诊断行为测试通过；8图smoke `RUN_FIT_DIAG_SMOKE_aed2ead55d2d40d78bd9234f421c90e5` 完成664条输出检查，正式Run使用同一源码和协议。[配置审计](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/CONFIG_AUDIT.json)通过；[回传后独立复算](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/LOCAL_RESULT_RECHECK.json)核对所有均值、区间、损伤逐GT计数、训练join及loss归约，和执行端一致。

Run成功完成与配置审计通过表示本诊断有效；`fit_diagnostic_gates`三项均为false，不能由诊断Run的`passed`恢复原S2配置晋级。

## 唯一下一步与推翻条件

启动前先冻结一项**零训练的监督—原生解码一致性审计**：沿用这14268个监督配对、同checkpoint与缓存，补测`proto网格、原监督ROI、logit>0`下的二值IoU，并与本次已保存的BCE和原图IoU逐对衔接。当前唯一缺失量是这一级二值IoU变化；只看BCE与最终RLE还无法定位首次分歧。

预定解释顺序：对已观察到的BCE下降/原图IoU下降配对，若proto同ROI二值IoU已下降，分歧首先出现在监督目标与二值质量之间；若此级改善而原图下降，后续仅核查原生解码及评价范围转换。各阶段数量如实报告，不能据一次平均值把所有实例归为同一原因。本次到此结案，下一项尚未执行，不调整loss、系数倍率或训练预算。

推翻条件：发现输入/最终检测身份、GT投影或评价实现错误时，撤销受影响的科学判定，保留原Run并执行独立纠错核验。当前配置和回传核验均通过。S1有限恢复机会结论仍保留；本研究继续在已接受的本地原生配置范围内解释，官方发表AP复现状态不因本诊断改变。

## 产物位置与停机

完整GT账本、固定配对表、loss表、原协议/源码快照等38项文件已回传当前Run目录并核验SHA；[transfer](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/transfer.json)、[回传核验](runs/RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926/LOCAL_TRANSFER_VERIFICATION.json)。80,632,865字节的B完整预测保留在笔记本 `D:\coco_wire\baseline_native_20261013\experiments\mask_readout_fit_diagnostic_20261014\runs\RUN_FIT_DIAG_FULL_1de2b1149b9d485eadf6afd98872c926\B_predictions.json`，SHA `c59d2e226b39ae8d5f548d7b60d05aca494094e0959605cac6990a8277f15e15`；原缓存仍在原Run，不复制或修改。

结束检查见 [EXECUTION_CLOSEOUT](EXECUTION_CLOSEOUT.json)：2026-10-14 00:39:50（+08:00）笔记本没有Python进程，诊断锁与原S2锁均不存在。
