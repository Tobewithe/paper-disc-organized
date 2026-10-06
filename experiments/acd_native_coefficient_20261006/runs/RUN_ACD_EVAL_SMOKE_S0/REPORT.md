# RUN_ACD_EVAL_SMOKE_S0 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 开始时间 | 未知 |
| 结束时间 | 2026-10-06T18:13:14.972886+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_EVAL_SMOKE_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_EVAL_SMOKE_S0`。
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
| `acd/predictions.json` | 本地未发现 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。

## 独立评价

summary status=complete；
图像=8；覆盖全部 annotation images=否。
范围：supplied COCO image subset; do not interpret as full-val or blind test。

COCO AP 均以百分数展示；差值为百分点。未知表示指标缺失、未回传或非有限值。

| 评价 | 模型 | AP50:95 | AP75 | APsmall |
|---|---|---:|---:|---:|
| Mask | official | 50.849 | 57.447 | 35.288 |
| Mask | baseline | 50.849 | 57.447 | 35.288 |
| Mask | acd | 未知 | 未知 | 未知 |
| Mask | ACD − baseline | 未知 | 未知 | 未知 |
| Box | official | 61.242 | 62.763 | 48.166 |
| Box | baseline | 61.242 | 62.763 | 48.166 |
| Box | acd | 未知 | 未知 | 未知 |
| Box | ACD − baseline | 未知 | 未知 | 未知 |

这里不提供 AP 置信区间；本实验的 bootstrap 只用于图像配对的 IoU/损伤诊断。

主要原始证据：
- [run.json](run.json)
- [SUMMARY.json](SUMMARY.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
