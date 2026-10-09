# QCR Stage I：已训练版本的完整评价与停止决定

记录日期：2026-10-06（Asia/Shanghai）。当前已训练 QCR 版本未建立 coefficient 纠错收益：FINAL 严格 MaskFail 目标组的主比较 D−B 为 **−1.5904 pp**，全体为 **−2.3638 pp**，两者图片 bootstrap 95% CI 均完全低于零；六项数值放行条件全部失败。**停止于 Stage I，不启动 Stage II，不扫超参、不扩大数据、不延长预算、不全模型微调。** 当前候选范围的完整 FINAL 统计无错误或缺项，协议合规性仍因实现偏差不完整。因此这支持停止当前版本，不能写成原协议设计完整合规的阴性复现，也不能推断所有可学习质量地形都无效。偏差与代码谱系详见 [AUDIT.md](AUDIT.md)。

本轮固定 epoch 3：A 为原始 c0，B 为 Direct32，D1 为 Q 第一梯度步，D 为已登记的两步 Q-Refine。原 YOLO 参数冻结，仅分别训练新增 encoder 与头；两臂 encoder 架构相同、权重独立，总参数分别 58,784 / 84,097。每臂实际遍历 3 次 selected-FIT：每 epoch 12,635 图、50,257 candidate、1,580 optimizer steps，总计 4,740 steps；Q 每实例 8 个状态，不能把相同更新数解释成相同计算量。入选 Failure / Success 为 20,257 / 30,000。历史目录名 seed0 保留，实际头训练 seed 为 **20261005**。半径 rho=12.0853271484375、lambda=.003、两步 eta=(rho/2,rho/4) 均固定，没有据 DEV/FINAL 调整。

FINAL 流式处理完 val2017 的 5,000 张图，未写全量 tensor cache，原模型 buffers 核验未变；36,335 个原图 noncrowd GT 中，36,213 个具有官方 one-to-one TAL 配对，122 个未配对。IoU 总体为这 36,213 行、4,952 张有候选图；48 张无候选图及未匹配 GT 不补零、不进入该 IoU 分母。严格目标组为 **5,304 行 / 2,228 图**：固定 TAL 的 class-correct、BoxIoU≥.75、baseline MaskIoU<.75，与“完整 argmax-correct raw 池有 Box75、无 Mask75”的交集；不覆盖未分配 GT 或所有 raw 位置，本轮不评价全 GT 召回、COCO AP 或输出后处理。旧 6,459 个 autoRect 诊断使用不同输入几何，不能拼接。DEV 历史缓存缺完整 raw 池，严格组保持未知。本项目已有先前 val 探索，本轮不能称首次 blind val。

评价使用完整 P@c、官方 process_mask 的 640 上采样/预测框裁剪/二值阈值，再反 letterbox 到原图，对独立 COCO annotation mask 计算正常 IoU。图片 macro 先平均同图有效候选再平均有效图片；CI 用 **5,000 次整图配对 bootstrap**，同图候选与各臂配对保留，空子组与未定义像素指标不填零。下表均为图片 macro，IoU 单位 %、差与 CI 单位 pp。

| 范围 | A | B | D | D−B [95% CI] | D−A [95% CI] | B−A [95% CI] |
|---|---:|---:|---:|---|---|---|
| DEV 全体（13,690 行 / 2,000 图） | 80.8471 | 80.2905 | 78.1244 | −2.1661 [−2.3445, −1.9861] | −2.7227 [−2.9074, −2.5430] | −0.5566 [−0.6412, −0.4717] |
| FINAL 全体 | 79.5417 | 78.9946 | 76.6308 | −2.3638 [−2.4892, −2.2401] | −2.9109 [−3.0461, −2.7771] | −0.5471 [−0.6032, −0.4930] |
| FINAL 严格目标组 | 60.9552 | 60.3055 | 58.7151 | −1.5904 [−1.9249, −1.2650] | −2.2400 [−2.6159, −1.8712] | −0.6496 [−0.8965, −0.3943] |

FINAL 第一步 D1 全体 IoU 为 71.4846%，比 A 下降 8.0572 pp。第二步部分恢复仍未超过 A 或 B，不能用“第二步比第一步好”作为方法放行依据。

| Q held-out 诊断（实例等权） | DEV | FINAL |
|---|---:|---:|
| Pairwise accuracy | 53.1602%（13,607 有效实例） | 53.0261%（35,920 有效实例；985,135 非平局对） |
| Spearman | 0.0820 | 0.0791 |
| 一步真实 IoU 严格上升 | 2,314 / 13,690 = 16.9028% | 6,423 / 36,213 = 17.7367% |
| 非零 ΔQ / ΔIoU 符号一致率 | 64.7433%（13,362） | 64.2211%（35,370） |
| ΔQ>0 却真实 IoU 下降 | 3,817 / 5,539 = 68.9114% | 10,093 / 14,727 = 68.5340% |

这些状态是训练后新增的、身份固定随机扰动诊断，未用 held-out oracle，不能追认为预锁定 oracle 方向测试。Pairwise 排除真实 hard-IoU 平局，Q 平局给 .5；Spearman 使用平均秩，常量状态未定义。一步改善分母保留 IoU 平局（FINAL 843）；符号一致率排除这些零增量。较高符号一致率主要包含共同下降，不能当作上升方向可靠性。C/Q-Rank 在固定候选上保留 c0，其 IoU 等于 A；**质量重排的 AP 效果未测**。

| FINAL 数值放行项 | 观察值 | 要求 | 结果 |
|---|---:|---:|---|
| Held-out pairwise | 53.0261% | ≥65% | 失败 |
| 一步真实 IoU 上升比例 | 17.7367% | >55% | 失败 |
| 严格组 D−B | −1.5904 pp | ≥+0.2 pp | 失败 |
| 严格组 D−B CI 下界 | −1.9249 pp | >0 | 失败 |
| 严格组 D−A | −2.2400 pp | ≥+0.5 pp | 失败 |
| 全体 D−A CI 下界 | −3.0461 pp | ≥−0.1 pp | 失败 |

Mask75 按每行 baseline <.75 / ≥.75 定义修复与损伤；全体这些分母不额外限制 class/box。严格目标组 baseline 均失败，只有修复率，不能据该子组评估保护成功实例的能力。

| 范围 / 方法 | 修复 / baseline失败 | 损伤 / baseline成功 | 净数量 | 图片净 Mask75 pp [95% CI] |
|---|---:|---:|---:|---|
| DEV B | 164 / 5,095 | 557 / 8,595 | −393 | −1.3612 [−1.7110, −0.9912] |
| DEV D | 267 / 5,095 | 1,407 / 8,595 | −1,140 | −6.1004 [−6.7563, −5.4506] |
| FINAL B | 510 / 14,129 | 1,435 / 22,084 | −925 | −1.5071 [−1.7576, −1.2559] |
| FINAL D | 740 / 14,129 | 3,895 / 22,084 | −3,155 | −6.1914 [−6.6354, −5.7463] |
| FINAL 严格组 B | 225 / 5,304 | 无成功分母 | +225 | +4.5267 [3.8115, 5.2242] |
| FINAL 严格组 D | 319 / 5,304 | 无成功分母 | +319 | +6.6706 [5.8328, 7.5466] |

D 修复更多，但全体损伤更大；严格组平均 IoU 仍下降。原图 coverage 是正常二值 mask 覆盖 GT 的比例；主 AUC 用反 letterbox 后连续裁剪 logits 在原图预测框支持内排序，FPR 用该支持内实际正常二值 mask 的非目标像素阳性率，其他实例也算非目标。未定义值保持 NA。下表采用已完成的原图像素 v2，均为图片 macro %；DEV TAL 失败为 2,550 行 / 981 图，不替代未知的 DEV 严格组。不可把 coverage 增加单独解读成更好掩码。

| 范围 / 方法 | Coverage | AUC | FPR |
|---|---:|---:|---:|
| DEV 全体 A / B / D | 92.3963 / 93.7744 / 92.4304 | 95.5142 / 95.5258 / 94.9895 | 20.9969 / 24.7590 / 28.0274 |
| DEV TAL 失败 A / B / D | 85.3681 / 89.1167 / 85.7231 | 90.1532 / 90.1309 / 89.3382 | 32.9777 / 40.6654 / 39.6379 |
| FINAL 全体 A / B / D | 91.6882 / 93.3069 / 91.7043 | 95.0747 / 95.0997 / 94.4910 | 22.0712 / 26.3045 / 29.4515 |
| FINAL 严格组 A / B / D | 83.0055 / 87.9043 / 84.3389 | 88.9929 / 89.0122 / 88.0090 | 33.4904 / 42.8883 / 41.7638 |

FINAL 全体 D−B 在 small / medium / large 分别为 −0.9716 / −2.5780 / −2.8210 pp，在 P3 / P4 / P5 分别为 −1.1386 / −2.3436 / −2.7671 pp，各自 CI 均低于零。尺度按原图 COCO area；原生层级 0/1/2 映射 P3/P4/P5 的修订只重新统计，不改原始行或重跑模型。全体层级计数为 11,330 / 14,099 / 10,784，严格组为 2,387 / 1,847 / 1,070；完整分层、像素增量与 CI 见 [FINAL 指标表](runs/RUN_stage1_final_metrics_repair_seed0/METRICS.md)。

FINAL correction 计时的 candidate mean 为 B **0.776 ms**、D **9.149 ms**：仅 encoder 与系数更新，排除 YOLO forward、mask 解码、GT、指标及统计；这是同步 CUDA、串行 batch 1 的额外开销，不是部署总延迟。

训练 rank 使用 soft-IoU 与幅差乘积 margin .05、Failure 随机半径使用全局 rho、图等权而非全局实例等权、缺部分显式输入，以及非近似 1:1 采样等偏差均已留在审计中，追加评价不能修复既有训练。FIT 存储抽样 10 图中 6 个 P 为 FP16 且缺 input_uint8，不能外推全量比例或宣称 lossless FP32 等价；DEV 全量评价记录 2,000 图 P 为 FP32，FINAL 5,000 图流式 FP32。

固定 DEV 清单的前十图 FP32 重算与[数值 witness](runs/RUN_stage1_dev_numeric_witness_seed0/NUMERIC_WITNESS.json)均已完成，覆盖 102 个 candidate。六个身份字段配对完整，P/h/c0/box/target 在 atol=rtol=3e−5 下全部通过；A 原图二值 mask 逐像素相同。原缓存与 [FP32 重算 Run](runs/RUN_stage1_dev_fp32_witness_seed0/)的 A/B/D1/D IoU、Mask75、三臂位移范数、Q0/Q1/Q2 与 held-out Q 值在全部 102 行逐项完全一致。这支持这些 DEV 样本的缓存与保存方法输出一致，不认证全量 FIT/DEV 数值等价，也不补救 FIT 量化或监督偏差。

主来源：[FINAL 统计修订 SUMMARY](runs/RUN_stage1_final_metrics_repair_seed0/SUMMARY.json)，原始执行 [RUN_stage1_final_complete_seed0_retry1](runs/RUN_stage1_final_complete_seed0_retry1/)；[DEV 统计修订 SUMMARY](runs/RUN_stage1_dev_metrics_repair_seed0/SUMMARY.json)及[完整指标表](runs/RUN_stage1_dev_metrics_repair_seed0/METRICS.md)，原图 v2 逐行来源 [RUN_stage1_dev_original_metrics_seed0](runs/RUN_stage1_dev_original_metrics_seed0/)。两次层级修订只改统计映射。训练为 [Direct retry5](runs/RUN_stage1_direct_seed0_retry5/) 与 [Quality](runs/RUN_stage1_quality_seed0/)；checkpoint SHA256 分别为 `25e0ac04be5398f989c3ad47f710b677e429e55a36799f794adc944d50200cc8` / `07796b8d5600827979073d167ea8e00236a6a287947f51962a6415730005c065`，协议 SHA256 为 `d8cb1deba60349c15afa0cbb6d9b37158cd5b02cd2bfebe0466df984186a27a7`。

[回传回执](assets/remote_receipts/TRANSFER_SUMMARY.json)记录 36 个 Run 目录、433 个选定文件已回传，run.json、实际源码快照、manifest.sha256 与 transfer.json 均保留；archive SHA256 为 `44100362295642cad7f40c9c23583e32193381ed25d32b609b7418961a8f5f80`。这是 selected-artifact 范围，epoch checkpoints 和完整 FIT/DEV tensor cache 仍在远端，未声称整目录镜像。模型执行、产物、传输和本地可用性分别记录。历史故障 Run、日志和部分产物均保留；训练 loss 下降不作为效用证据。

[独立核验](assets/remote_receipts/VERIFICATION.json)记录 errors=[]：433 个选定文件、155 个源码快照及 archive SHA 全部吻合，FINAL/DEV 身份唯一性、逐图数量与候选合计、逐行官方解码零像素差均通过，层级统计修订完整保留主 IoU/CI 且 size/P3–P5 分组合计正确；该核验不等于科学协议放行。DEV converted-GT 索引范围的离线记录仍为 unknown，不以普通 GT 数量补填。核验记录 SHA256 为 `568681c24a2d2475a646b06fb164c879bd03e36b6b9b562c1edaadf467f5f7fa`。
