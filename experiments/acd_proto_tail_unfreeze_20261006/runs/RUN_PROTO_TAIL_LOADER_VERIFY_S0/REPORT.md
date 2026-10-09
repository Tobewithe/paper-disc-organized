# RUN_PROTO_TAIL_LOADER_VERIFY_S0 报告

生成于 2026-10-06T23:29:37+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 包裹层开始时间 | 未知 |
| 包裹层结束时间 | 未知 |
| 远端实际目录 | D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_LOADER_VERIFY_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_LOADER_VERIFY_S0`。
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

run.json 的待验证项：Exact wrapper start/end timestamps unavailable after metadata reconciliation。

## 元数据恢复与仍未知的时间

Copying planned Run templates overwrote diagnostic run.json; observed zero exit codes and preserved PASS receipts restore status, not unknown wrapper timestamps.
诊断状态依据 observed exit code 与 preserved PASS receipt 恢复；精确 wrapper 开始/结束仍保持未知。
receipt 自身诊断开始=未知；诊断完成=2026-10-06T21:55:37.759354+08:00（Asia/Shanghai）。

## 独立诊断 receipt

[LOADER_VERIFICATION.json](LOADER_VERIFICATION.json)：status=PASS。
这是梯度/optimizer/EMA 或 synthetic loader 合同诊断；不是正式三轮训练、不是 AP 结果。
receipt 开始=未知；结束=2026-10-06T21:55:37.759354+08:00（Asia/Shanghai）。
包裹层时间和 receipt 时间分别保留，缺失保持未知。

主要原始证据：
- [run.json](run.json)
- [LOADER_VERIFICATION.json](LOADER_VERIFICATION.json)
- [SOURCE.json](SOURCE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)

## 实际执行源码与协议快照

SOURCE.json 记录的执行版本：`D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006\scripts\verify_loader.py`；记录时间=2026-10-06T21:55:33.610300+08:00（Asia/Shanghai）。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/verify_loader.py) | `70ffb39dd54684b50c32a2ccdee2ad4cced6ee793f50650c532ea34a934338e0` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。

执行方式记录：Root driver executed with a byte-identical contemporaneous source snapshot; evaluator imported from Run/source
