# RUN_BASELINE_SMOKE_S0_RETRY1 报告

生成于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 开始时间 | 2026-10-06T18:17:58.534692+08:00 |
| 结束时间 | 2026-10-06T18:18:31.742285+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_native_coefficient_20261006/runs/RUN_BASELINE_SMOKE_S0_RETRY1 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_native_coefficient_20261006/runs/RUN_BASELINE_SMOKE_S0_RETRY1`。
transfer.json：可读；manifest.sha256：存在。

| 关键文件 | 当前本地可用性 |
|---|---|
| `TRAINING_COMPLETE.json` | 存在 |
| `TRAINING_AUDIT.json` | 存在 |
| `optimizer_step_audit.jsonl` | 本地未发现 |
| `input_batch_hashes.jsonl` | 存在 |
| `coeff_final.pt` | 存在 |
| `coeff_final_ema.pt` | 存在 |
| `trainer/weights/last.pt` | 本地未发现 |
| `SUMMARY.json` | 本地未发现 |

本地未发现不代表远端未生成；远端生成/执行事实以 receipt 为证据，回传状态以 transfer 记录为证据。

这是独立重试 Run：RUN_BASELINE_SMOKE_S0_RETRY1；它不覆盖原尝试的状态或路径。

## 训练审计

TRAINING_COMPLETE receipt：已存在；
receipt arm=baseline，epoch=1，
seed=0，Ultralytics=8.4.100。

有限 loss：通过；
frozen 参数及 BN buffers 不变：通过；
transformed-input batches=16。

梯度数值的记录单位：AMP-scaled before optimizer unscale。

原始 AMP-scaled 梯度、unscale 后/clip 前观测梯度、实际 optimizer 调用处的梯度分别记录。
观察到 Inf/NaN 的 attempt 保留为非有限；只有被 GradScaler 实际跳过的 attempt 才可排除出 applied 集合。
实际应用梯度 gate：未知。

| optimizer / GradScaler 全局审计 | 数值 |
|---|---|
| 尝试更新 step | 未知 |
| 实际应用 step | 未知 |
| 确认 AMP overflow skip step | 未知 |
| 官方 EMA updates（包括 overflow-skip attempt） | 未知 |
| unscale 后/clip 前梯度全部有限 | 未知 |
| 原始 unscaled 非有限 attempt | 未知 |
| 非有限且确认由 scaler 跳过的 attempt | 未知 |
| 所有原始非有限 attempt 均确认被跳过 | 未知 |
| 实际应用梯度全部有限 | 未知 |
| 危险 optimizer 调用被拦截次数 | 未知 |
| 未分类 skip 次数 | 未知 |

此 Run 使用旧梯度审计，缺少直接 optimizer invocation / GradScaler skip 记录。
旧 `all_finite` 记录的是 AMP-scaled 梯度，不能转写成实际应用梯度有限；
旧审计的 `all_finite=false` 保持为 false。实际 applied/skip 数量未知，末端 scaler scale 不能证明精确跳步数量。

| coefficient branch | 更新参数数 | scaled 全部有限 | scaled 非有限 attempts | unscaled 全部有限 | unscaled 非有限 attempts | 实际应用全部有限 | 实际应用 steps | 实际应用非零 steps | 应用梯度及更新 gate |
|---|---:|---|---:|---|---:|---|---:|---:|---|
| one2many | 12 | 是 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 |
| one2one | 12 | 是 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 | 未知 |

| ACD loss branch | loss calls | candidates | accepted actions | FP pixel terms | aux sum |
|---|---:|---:|---:|---:|---:|
| one2many | 0 | 0 | 0 | 0 | 0.0000 |
| one2one | 0 | 0 | 0 | 0 | 0.0000 |

baseline 关闭 action 项；其 accepted actions=0 不代表训练失败。ACD 的候选/接受数分别保留两个 loss branch 的实际记录。
冻结审计针对 live tensors；最终评价必须用 FP32 EMA coefficient 文件覆盖原始官方 FP32 模型。

## 内部末 epoch 指标

这些指标来自训练时的转换标签验证范围；smoke/196 张内部验证指标与原始 COCO 全 5,000 张独立评价范围不同。

```json
{
  "metrics/precision(B)": 0.7095523267486604,
  "metrics/recall(B)": 0.7313094508140611,
  "metrics/mAP50(B)": 0.8861390124151733,
  "metrics/mAP50-95(B)": 0.6566728312416857,
  "metrics/precision(M)": 0.6742968300266139,
  "metrics/recall(M)": 0.6786778718666927,
  "metrics/mAP50(M)": 0.7998524627075709,
  "metrics/mAP50-95(M)": 0.5247087357542698
}
```

主要原始证据：
- [run.json](run.json)
- [TRAINING_SETUP.json](TRAINING_SETUP.json)
- [TRAINING_AUDIT.json](TRAINING_AUDIT.json)
- [TRAINING_COMPLETE.json](TRAINING_COMPLETE.json)
- [transfer.json](transfer.json)
- [manifest.sha256](manifest.sha256)
- [stdout.log](stdout.log)
