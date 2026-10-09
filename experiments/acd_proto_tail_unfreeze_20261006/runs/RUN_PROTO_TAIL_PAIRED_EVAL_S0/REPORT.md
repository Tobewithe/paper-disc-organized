# RUN_PROTO_TAIL_PAIRED_EVAL_S0 报告

生成于 2026-10-06T23:29:37+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 包裹层开始时间 | 2026-10-06T22:46:19.308440+08:00 |
| 包裹层结束时间 | 2026-10-06T23:27:20.700631+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PAIRED_EVAL_S0`。
transfer.json：可读；manifest.sha256：存在。

| 关键文件 | 当前本地可用性 |
|---|---|
| `TRAINING_COMPLETE.json` | 本地未发现 |
| `TRAINING_AUDIT.json` | 本地未发现 |
| `optimizer_step_audit.jsonl` | 本地未发现 |
| `input_batch_hashes.jsonl` | 本地未发现 |
| `mask_final.pt` | 本地未发现 |
| `mask_final_ema.pt` | 本地未发现 |
| `trainer/weights/last.pt` | 本地未发现 |
| `SUMMARY.json` | 存在 |
| `official/predictions.json` | 存在 |
| `baseline/predictions.json` | 存在 |
| `acd/predictions.json` | 存在 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。

## 独立评价

summary status=complete；
图像=5000；覆盖全部 annotation images=是。
范围：all images in supplied original COCO annotations。

COCO AP 以百分数展示，差值为百分点；未知表示缺失、未回传或非有限值。

| 评价 | 模型/比较 | AP50:95 | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---|---:|---:|---:|---:|---:|---:|
| Mask | official | 43.685 | 66.354 | 47.295 | 22.402 | 47.670 | 63.572 |
| Mask | baseline | 43.445 | 66.323 | 47.054 | 22.387 | 47.460 | 63.071 |
| Mask | acd | 43.488 | 66.386 | 47.036 | 22.446 | 47.523 | 63.188 |
| Mask | acd − baseline | +0.043 | +0.063 | -0.019 | +0.060 | +0.063 | +0.117 |
| Mask | acd − official | -0.198 | +0.033 | -0.259 | +0.045 | -0.147 | -0.384 |
| Mask | baseline − official | -0.241 | -0.031 | -0.240 | -0.015 | -0.210 | -0.501 |
| Box | official | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | baseline | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | acd | 52.116 | 69.141 | 56.935 | 33.955 | 57.012 | 68.020 |
| Box | acd − baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Box | acd − official | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| Box | baseline − official | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

不提供 AP 置信区间；bootstrap 仅用于图像簇配对的 IoU/损伤诊断。
原型尾部学习和 baseline 普通微调都可能改变 Mask AP；ACD − official 也保留以显示微调是否退化。

## 实例配对诊断

配对定义：每个 baseline detection 以同类 box IoU ≥ 0.5 匹配最高重叠的 noncrowd GT；
baseline 原图 mask IoU ≥ 0.75 为成功，ACD 降至 < 0.75 为损伤，反向越过该阈值为修复。
允许多个 detection 匹配同一 GT，计数是 detection identity 的诊断，不能当成 COCO AP 一对一匹配。

配对可用性：通过。

| 计数 | 数值 |
|---|---:|
| 配对匹配 detections | 86600 |
| baseline-success | 35652 |
| 成功组损伤 | 207 |
| baseline-failure | 50948 |
| 失败组修复 | 240 |
| 评价图像 | 5000 |
| 含匹配 detection 的图像 | 4946 |
| unique matched GT | 33644 |

| 图像簇配对诊断 | 点估计 | 95% bootstrap 区间 |
|---|---:|---:|
| 所有匹配 detection 平均 mask IoU 变化 | -0.000313 | [-0.000369, -0.000255] |
| baseline-success 平均 mask IoU 变化 | 0.000216 | [0.000177, 0.000257] |
| baseline-success 损伤率（%） | 0.5806 | [0.4998, 0.6599] |
| baseline-failure 修复率（%） | 0.4711 | [0.4104, 0.5359] |
| baseline 平均 mask IoU | 0.662249 | [0.658890, 0.665579] |
| ACD 平均 mask IoU | 0.661936 | [0.658553, 0.665272] |

bootstrap：5000 次，seed=0；
单位为图像簇，两臂配对抽样，以簇总和之比统计。记录定义：supplied images, paired across arms; ratio of clustered sums。
区间描述当前固定模型与该验证集的诊断变化，不量化不同训练 seed 的不确定性。

## 预定义可行性判定

结论：**本次可行性门槛未通过**。有效性核验通过，但至少一个预定义科学阈值未通过；保留本次未达门槛的结果并停止调参。

主比较为 ACD − 同范围 coefficient+proto-tail baseline；official 为额外参考。
negative 表示未达到预设门槛，不能据此断言方法整体无效。

| 协议 gate | 判定 |
|---|---|
| 协议锁 | 通过 |
| baseline 固定训练预算/版本/数据/BN | 通过 |
| baseline 四个 trainable prefixes/2,306,240 参数/全 BN 冻结 | 通过 |
| baseline 有限 live loss | 通过 |
| baseline 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| baseline frozen 参数及 BN buffers 不变 | 通过 |
| baseline one2many 实际应用梯度有限且每个 prefix 更新 | 通过 |
| baseline one2one 实际应用梯度有限且每个 prefix 更新 | 通过 |
| baseline proto_tail 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd 固定训练预算/版本/数据/BN | 通过 |
| acd 四个 trainable prefixes/2,306,240 参数/全 BN 冻结 | 通过 |
| acd 有限 live loss | 通过 |
| acd 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| acd frozen 参数及 BN buffers 不变 | 通过 |
| acd one2many 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd one2one 实际应用梯度有限且每个 prefix 更新 | 通过 |
| acd proto_tail 实际应用梯度有限且每个 prefix 更新 | 通过 |
| 实际模型 one2many 原型尾部梯度/one2one detach/官方单步诊断 | 通过 |
| 完整 prototype-tail FP32 overlay 的独立 loader 合同 | 通过 |
| 正式训练/评估/诊断已完整回传且原 source manifest 保留 | 通过 |
| 配对 transformed-input 完整流一致 | 通过 |
| 两臂 optimizer attempt 的 batch/LR/accumulate/AMP schedule 一致 | 通过 |
| DATA_RECEIPT 与 fit/internal-val/smoke/full-val 列表哈希及数量 | 通过 |
| 完整 5,000 张原始 COCO val2017 独立评估 | 通过 |
| 固定第 3 epoch EMA 四-prefix mask-state overlay 官方 FP32 推理 | 通过 |
| official/baseline/acd frozen FP32 tensors digest 一致 | 通过 |
| ACD/baseline box/class/confidence/detection identity 直接逐图一致 | 通过 |
| Mask AP：ACD − baseline >= +0.003 | 未通过 |
| Mask AP75/APsmall 不同时下降 | 通过 |
| 配对损伤诊断有 baseline-success 样本且图像 bootstrap 完整 | 通过 |
| baseline-success 损伤率 <= 1% | 通过 |
| baseline-success 平均 mask IoU 变化 >= −0.005 | 通过 |

未知不能当成通过。执行或 frozen/data/parity 核验无效时，不将 AP 差解释为训练迁移结论。
固定预算结束后停止；不自动加 seed、修改 auxiliary 设置或扩展训练。

## 配对训练数值轨迹

| 正式训练臂 | 选用 Run | attempted | applied | confirmed overflow skips | observed unscaled nonfinite attempts | EMA updates |
|---|---|---:|---:|---:|---:|---:|
| baseline | RUN_PROTO_TAIL_BASELINE_FEASIBILITY_S0 | 69 | 66 | 3 | 3 | 69 |
| acd | RUN_PROTO_TAIL_ACD_FEASIBILITY_S0 | 69 | 66 | 3 | 3 | 69 |

overflow-skip 次数差（ACD − baseline）：+0。
已直接核验两臂 overflow-skip 的 attempt、epoch 和 batch 位置相同；损失与梯度值仍由各自训练条件决定。
非有限梯度观测不改称 finite；计数差异不自动归因于 ACD。
attempt 的 epoch/batch/LR/accumulate/AMP schedule 必须一致；applied/overflow-skip 数量允许按官方 AMP 行为分别记录。
EMA 保持官方时机：每次 optimizer attempt 都更新，包括 scaler 跳过 actual optimizer step 的 attempt。

完整判定及来源 SHA256：`DECISION.json`。

主要原始证据：
- [run.json](run.json)
- [SUMMARY.json](SUMMARY.json)
- [SOURCE.json](SOURCE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
- [stdout.log](stdout.log)

## 实际执行源码与协议快照

SOURCE.json 记录的执行版本：`D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_PAIRED_EVAL_S0\source\evaluate.py`；记录时间=2026-10-06T22:46:19.306435+08:00（Asia/Shanghai）。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/evaluate.py) | `82aa72bf536d570db1c763036bbf8b74a649d7bfd472f7cf8de2cee4114f5ada` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。
