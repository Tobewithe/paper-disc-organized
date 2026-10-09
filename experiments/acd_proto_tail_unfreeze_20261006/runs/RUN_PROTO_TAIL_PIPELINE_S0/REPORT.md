# RUN_PROTO_TAIL_PIPELINE_S0 报告

生成于 2026-10-06T23:29:37+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 包裹层开始时间 | 2026-10-06T21:58:52.324229+08:00 |
| 包裹层结束时间 | 2026-10-06T23:27:20.705885+08:00 |
| 远端实际目录 | D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006 |
| 解释器 | C:\Users\28358\anaconda3\envs\pytorch\python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_PIPELINE_S0`。
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
| `SUMMARY.json` | 本地未发现 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。

## Pipeline 进度

```json
{
  "stage": "RUN_PROTO_TAIL_PAIRED_EVAL_S0",
  "updated_at": "2026-10-06T22:46:19.311467+08:00"
}
```

Pipeline COMPLETE receipt：

```json
{
  "completed": true,
  "ended_at": "2026-10-06T23:27:20.703638+08:00",
  "scope": "Locked smoke, paired three-epoch training and full val2017 evaluation"
}
```

当前未发现完整训练/评价结果 receipt；任何未回传结果保持未知。

主要原始证据：
- [run.json](run.json)
- [SOURCE.json](SOURCE.json)
- [OPTIMIZER_SCHEDULE_PARITY.json](OPTIMIZER_SCHEDULE_PARITY.json)
- [PROGRESS.json](PROGRESS.json)
- [COMPLETE.json](COMPLETE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
- [stdout.log](stdout.log)

## 实际执行源码与协议快照

SOURCE.json 记录的执行版本：`D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_PIPELINE_S0\source\run_pipeline.py`；记录时间=2026-10-06T21:58:50.925361+08:00（Asia/Shanghai）。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/run_pipeline.py) | `4f9113469409c7acafe4e14f8d7bb1a2d6c45df46edf5b2c9aef607de6b06809` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。
