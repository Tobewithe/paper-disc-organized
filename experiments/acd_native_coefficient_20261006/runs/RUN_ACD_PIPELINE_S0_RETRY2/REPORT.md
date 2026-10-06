# RUN_ACD_PIPELINE_S0_RETRY2 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 开始时间 | 2026-10-06T18:56:25.377801+08:00 |
| 结束时间 | 2026-10-06T20:17:35.57546+08:00 |
| 远端实际目录 | D:\coco_wire\experiments\acd_native_coefficient_20261006 |
| 解释器 | C:\Users\28358\anaconda3\envs\pytorch\python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_PIPELINE_S0_RETRY2`。
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

这是独立重试 Run：RUN_ACD_PIPELINE_S0_RETRY2；它不覆盖原尝试的状态或路径。

## Pipeline 进度

```json
{
  "stage": "RUN_ACD_PAIRED_EVAL_S0",
  "updated_at": "2026-10-06T19:36:39.068057+08:00"
}
```

Pipeline COMPLETE receipt：

```json
{
  "completed": true,
  "ended_at": "2026-10-06T20:17:35.573973+08:00",
  "scope": "Locked smoke, paired three-epoch training and full val2017 evaluation"
}
```

当前未发现完整训练/评价结果 receipt；任何未回传结果保持未知。

主要原始证据：
- [run.json](run.json)
- [OPTIMIZER_SCHEDULE_PARITY.json](OPTIMIZER_SCHEDULE_PARITY.json)
- [PROGRESS.json](PROGRESS.json)
- [COMPLETE.json](COMPLETE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
- [stdout.log](stdout.log)
