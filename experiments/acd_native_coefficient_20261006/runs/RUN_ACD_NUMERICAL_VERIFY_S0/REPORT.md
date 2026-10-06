# RUN_ACD_NUMERICAL_VERIFY_S0 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | failed |
| 执行状态 | failed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 1 |
| 开始时间 | 未知 |
| 结束时间 | 未知 |
| 远端实际目录 | D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_NUMERICAL_VERIFY_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_NUMERICAL_VERIFY_S0`。
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
| `SUMMARY.json` | 本地未发现 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。
## 失败/重试事实

失败记录：verify_optimizer_audit mock counter omitted an initialized zero field: KeyError('unclassified_skipped_steps'); no detector or COCO training in this Run。
失败及部分产物保留；未观测的退出码或原因保持未知。

当前未发现完整训练/评价结果 receipt；任何未回传结果保持未知。

主要原始证据：
- [run.json](run.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
