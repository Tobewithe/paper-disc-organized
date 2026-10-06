# RUN_ACD_PAIRED_EVAL_S0 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 开始时间 | 2026-10-06T19:36:39.065962+08:00 |
| 结束时间 | 2026-10-06T20:17:35.559569+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_PAIRED_EVAL_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_PAIRED_EVAL_S0`。
transfer.json：可读；manifest.sha256：存在。

| 关键文件 | 当前本地可用性 |
|---|---|
| `TRAINING_COMPLETE.json` | 本地未发现 |
| `TRAINING_AUDIT.json` | 本地未发现 |
| `optimizer_step_audit.jsonl` | 本地未发现 |
| `input_batch_hashes.jsonl` | 本地未发现 |
| `coeff_final.pt` | 本地未发现 |
| `coeff_final_ema.pt` | 本地未发现 |
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

COCO AP 均以百分数展示；差值为百分点。未知表示指标缺失、未回传或非有限值。

| 评价 | 模型 | AP50:95 | AP75 | APsmall |
|---|---|---:|---:|---:|
| Mask | official | 43.685 | 47.295 | 22.402 |
| Mask | baseline | 43.577 | 47.198 | 22.417 |
| Mask | acd | 43.614 | 47.282 | 22.522 |
| Mask | ACD − baseline | +0.037 | +0.084 | +0.105 |
| Box | official | 52.116 | 56.935 | 33.955 |
| Box | baseline | 52.116 | 56.935 | 33.955 |
| Box | acd | 52.116 | 56.935 | 33.955 |
| Box | ACD − baseline | +0.000 | +0.000 | +0.000 |

这里不提供 AP 置信区间；本实验的 bootstrap 只用于图像配对的 IoU/损伤诊断。

## 实例配对诊断

配对定义：每个 baseline detection 以同类 box IoU ≥ 0.5 匹配最高重叠的 noncrowd GT；
baseline 原图 mask IoU ≥ 0.75 为成功，ACD 降至 < 0.75 为损伤，反向越过该阈值为修复。
允许多个 detection 匹配同一 GT，计数是 detection identity 的诊断，不能当成 COCO AP 一对一匹配。

配对可用性：通过。

| 计数 | 数值 |
|---|---:|
| 配对匹配 detections | 86600 |
| baseline-success | 35916 |
| 成功组损伤 | 145 |
| baseline-failure | 50684 |
| 失败组修复 | 197 |
| 评价图像 | 5000 |
| 含匹配 detection 的图像 | 4946 |
| unique matched GT | 33644 |

| 图像簇配对诊断 | 点估计 | 95% bootstrap 区间 |
|---|---:|---:|
| 所有匹配 detection 平均 mask IoU 变化 | -0.000487 | [-0.000536, -0.000434] |
| baseline-success 平均 mask IoU 变化 | 0.000103 | [0.000073, 0.000134] |
| baseline-success 损伤率（%） | 0.4037 | [0.3393, 0.4719] |
| baseline-failure 修复率（%） | 0.3887 | [0.3338, 0.4438] |
| baseline 平均 mask IoU | 0.662233 | [0.658780, 0.665645] |
| ACD 平均 mask IoU | 0.661746 | [0.658278, 0.665153] |

bootstrap：5000 次，seed=0；
单位为图像簇，两臂配对抽样，以簇总和之比统计。记录定义：supplied images, paired across arms; ratio of clustered sums。
区间描述当前固定模型与该验证集的诊断变化，不量化不同训练 seed 的不确定性。

## 预定义可行性判定

结论：**本次可行性门槛未通过**。有效性核验通过，但至少一个预定义科学阈值未通过；保留本次未达门槛的结果并停止调参。

主比较为 ACD − coefficient-only baseline，official 为额外参考。
negative 表示未达到预设门槛，不能据此断言方法整体无效。

| 协议 gate | 判定 |
|---|---|
| 协议锁 | 通过 |
| baseline 固定训练预算/版本/数据/BN | 通过 |
| baseline 有限 live loss | 通过 |
| baseline 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| baseline frozen 参数及 BN buffers 不变 | 通过 |
| baseline one2many 实际应用梯度有限且系数更新 | 通过 |
| baseline one2one 实际应用梯度有限且系数更新 | 通过 |
| acd 固定训练预算/版本/数据/BN | 通过 |
| acd 有限 live loss | 通过 |
| acd 实际应用梯度有限且非有限 attempt 均确认由 GradScaler 跳过 | 通过 |
| acd frozen 参数及 BN buffers 不变 | 通过 |
| acd one2many 实际应用梯度有限且系数更新 | 通过 |
| acd one2one 实际应用梯度有限且系数更新 | 通过 |
| 配对 transformed-input 完整流一致 | 通过 |
| 两臂 optimizer attempt 的 batch/LR/accumulate/AMP schedule 一致 | 通过 |
| 完整 5,000 张原始 COCO val2017 独立评估 | 通过 |
| 固定第 3 epoch EMA coefficient overlay 官方 FP32 推理 | 通过 |
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
| baseline | RUN_BASELINE_FEASIBILITY_S0_RETRY1 | 69 | 66 | 3 | 3 | 69 |
| acd | RUN_ACD_FEASIBILITY_S0 | 69 | 66 | 3 | 3 | 69 |

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

执行 snapshot：`D:\coco_wire\experiments\acd_native_coefficient_20261006\runs\RUN_ACD_PAIRED_EVAL_S0\source\evaluate.py`；记录时间=2026-10-06T19:36:39.063939+08:00。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/evaluate.py) | `1f9b765b782e6b1c78f05cf10cc175a472ef0101141214cc3c00e2afab8d9c3b` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `18db322c67761ea0cd0400f5a518ba3c958ca62d01f7aa86d87ad34fb1b4971d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。
