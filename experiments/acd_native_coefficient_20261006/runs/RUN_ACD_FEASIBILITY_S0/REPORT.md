# RUN_ACD_FEASIBILITY_S0 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 开始时间 | 2026-10-06T19:14:15.461108+08:00 |
| 结束时间 | 2026-10-06T19:36:38.993877+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_FEASIBILITY_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_ACD_FEASIBILITY_S0`。
transfer.json：可读；manifest.sha256：存在。

| 关键文件 | 当前本地可用性 |
|---|---|
| `TRAINING_COMPLETE.json` | 存在 |
| `TRAINING_AUDIT.json` | 存在 |
| `optimizer_step_audit.jsonl` | 存在 |
| `input_batch_hashes.jsonl` | 存在 |
| `coeff_final.pt` | 存在 |
| `coeff_final_ema.pt` | 存在 |
| `trainer/weights/last.pt` | 本地未发现 |
| `SUMMARY.json` | 本地未发现 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。

## 训练审计

TRAINING_COMPLETE receipt：已存在；
receipt arm=acd，epoch=3，
seed=0，Ultralytics=8.4.100。

有限 loss：通过；
frozen 参数及 BN buffers 不变：通过；
transformed-input batches=1194。

梯度数值的记录单位：unscaled after official clip, at actual optimizer invocation。

原始 AMP-scaled 梯度、unscale 后/clip 前观测梯度、实际 optimizer 调用处的梯度分别记录。
观察到 Inf/NaN 的 attempt 保留为非有限；只有被 GradScaler 实际跳过的 attempt 才可排除出 applied 集合。
实际应用梯度 gate：通过。

| optimizer / GradScaler 全局审计 | 数值 |
|---|---|
| 尝试更新 step | 69 |
| 实际应用 step | 66 |
| 确认 AMP overflow skip step | 3 |
| 官方 EMA updates（包括 overflow-skip attempt） | 69 |
| unscale 后/clip 前梯度全部有限 | 否 |
| 原始 unscaled 非有限 attempt | 3 |
| 非有限且确认由 scaler 跳过的 attempt | 3 |
| 所有原始非有限 attempt 均确认被跳过 | 是 |
| 实际应用梯度全部有限 | 是 |
| 危险 optimizer 调用被拦截次数 | 0 |
| 未分类 skip 次数 | 0 |

| coefficient branch | 更新参数数 | scaled 全部有限 | scaled 非有限 attempts | unscaled 全部有限 | unscaled 非有限 attempts | 实际应用全部有限 | 实际应用 steps | 实际应用非零 steps | 应用梯度及更新 gate |
|---|---:|---|---:|---|---:|---|---:|---:|---|
| one2many | 12 | 否 | 2 | 否 | 2 | 是 | 66 | 66 | 通过 |
| one2one | 12 | 否 | 1 | 否 | 1 | 是 | 66 | 66 | 通过 |

| ACD loss branch | loss calls | candidates | accepted actions | FP pixel terms | aux sum |
|---|---:|---:|---:|---:|---:|
| one2many | 1194 | 28430 | 4923 | 1984605 | 1195.0530 |
| one2one | 1194 | 23851 | 4182 | 3287409 | 1458.4579 |

baseline 关闭 action 项；其 accepted actions=0 不代表训练失败。ACD 的候选/接受数分别保留两个 loss branch 的实际记录。
冻结审计针对 live tensors；最终评价必须用 FP32 EMA coefficient 文件覆盖原始官方 FP32 模型。

## 内部末 epoch 指标

这些指标来自训练时的转换标签验证范围；smoke/196 张内部验证指标与原始 COCO 全 5,000 张独立评价范围不同。

```json
{
  "metrics/precision(B)": 0.7317463804799142,
  "metrics/recall(B)": 0.677877587247957,
  "metrics/mAP50(B)": 0.7569283156729928,
  "metrics/mAP50-95(B)": 0.600519874343309,
  "metrics/precision(M)": 0.7164662812793985,
  "metrics/recall(M)": 0.6533924432085859,
  "metrics/mAP50(M)": 0.7071141971791581,
  "metrics/mAP50-95(M)": 0.4879772204581241
}
```

主要原始证据：
- [run.json](run.json)
- [TRAINING_SETUP.json](TRAINING_SETUP.json)
- [TRAINING_AUDIT.json](TRAINING_AUDIT.json)
- [TRAINING_COMPLETE.json](TRAINING_COMPLETE.json)
- [optimizer_step_audit.jsonl](optimizer_step_audit.jsonl)
- [SOURCE.json](SOURCE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
- [stdout.log](stdout.log)

## 实际执行源码与协议快照

执行 snapshot：`D:\coco_wire\experiments\acd_native_coefficient_20261006\runs\RUN_ACD_FEASIBILITY_S0\source\train_acd.py`；记录时间=2026-10-06T19:14:15.457688+08:00。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/train_acd.py) | `f95749d1ed34283d6ab4f3f9ab44b2bbbda0b081434826ffaa81a33b71732e4d` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `18db322c67761ea0cd0400f5a518ba3c958ca62d01f7aa86d87ad34fb1b4971d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。
