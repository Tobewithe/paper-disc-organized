# RUN_PROTO_TAIL_ACD_FEASIBILITY_S0 报告

生成于 2026-10-06T23:29:37+08:00（Asia/Shanghai）。

本报告仅依据当前本地可读文件；执行、产物、回传、文件可用性分别记录。

| 记录项 | 当前值 |
|---|---|
| Run status | completed |
| 执行状态 | completed |
| 产物状态 | available |
| 回传状态 | verified |
| exit code | 0 |
| 包裹层开始时间 | 2026-10-06T22:19:00.715545+08:00 |
| 包裹层结束时间 | 2026-10-06T22:46:19.236787+08:00 |
| 远端实际目录 | D:/coco_wire/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_ACD_FEASIBILITY_S0 |
| 解释器 | C:/Users/28358/anaconda3/envs/pytorch/python.exe |

本地 Run 目录：`C:/Dpan/codexproject/paper-disc-organized/experiments/acd_proto_tail_unfreeze_20261006/runs/RUN_PROTO_TAIL_ACD_FEASIBILITY_S0`。
transfer.json：可读；manifest.sha256：存在。

| 关键文件 | 当前本地可用性 |
|---|---|
| `TRAINING_COMPLETE.json` | 存在 |
| `TRAINING_AUDIT.json` | 存在 |
| `optimizer_step_audit.jsonl` | 存在 |
| `input_batch_hashes.jsonl` | 存在 |
| `mask_final.pt` | 存在 |
| `mask_final_ema.pt` | 存在 |
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
| one2many | 12 | 是 | 0 | 是 | 0 | 是 | 66 | 66 | 通过 |
| one2one | 12 | 是 | 0 | 是 | 0 | 是 | 66 | 66 | 通过 |
| proto_tail | 2 | 否 | 3 | 否 | 3 | 是 | 66 | 66 | 通过 |

| ACD loss branch | loss calls | candidates | accepted actions | FP pixel terms | aux sum |
|---|---:|---:|---:|---:|---:|
| one2many | 1194 | 28430 | 4663 | 2177078 | 1158.2468 |
| one2one | 1194 | 23851 | 4136 | 3235225 | 1456.2146 |

baseline 关闭 action 项；其 accepted actions=0 不代表训练失败。ACD 的候选/接受数分别保留两个 loss branch 的实际记录。
proto_tail 是参数/梯度组，不是第三个 ACD loss branch；one2one prototype 保持 detach。
冻结审计覆盖 live/EMA 所有其它 state 与所有 BN；最终评价用 FP32 EMA coefficient+proto-tail state 覆盖原始官方 FP32 模型。

## 可训练范围及每个 prefix 的实际变化

参数总数：实际=2306240，官方 m 预期=2306240。

| 参数组 | prefix | changed parameters | 该 prefix 预期完整 overlay state keys |
|---|---|---:|---:|
| one2many | `model.23.cv4.` | 12 | 42 |
| one2one | `model.23.one2one_cv4.` | 12 | 42 |
| proto_tail | `model.23.proto.cv2.` | 1 | 6 |
| proto_tail | `model.23.proto.cv3.` | 1 | 6 |

正式 payload 四 prefix 共 96 个 state keys：26 个可变 non-BN 参数与 70 个原始冻结 BN state；
训练 count 和 names 与 loader 的 exact keys/dtype/BN 原始值合同分别核验。

| 初始 optimizer group | LR | Muon | 参数数 | proto-tail prefixes |
|---|---:|---|---:|---|
| 0 / weight | 0.00114000 | 否 | 0 |  |
| 1 / weight | 0.00038000 | 否 | 0 |  |
| 2 / bn | 0.00114000 | 否 | 0 |  |
| 3 / bn | 0.00038000 | 否 | 0 |  |
| 4 / bias | 0.00114000 | 否 | 0 |  |
| 5 / bias | 0.00038000 | 否 | 6 |  |
| 6 / muon | 0.00114000 | 是 | 1 | model.23.proto.cv3. |
| 7 / muon | 0.00038000 | 是 | 19 | model.23.proto.cv2. |

保留官方 MuSGD 名称规则：prototype cv3 获得 3× group LR，cv2 为普通 group LR。
官方全参数 clip_grad_norm_(..., 10) 的联合范数会随可训练原型尾部增大；可能影响系数梯度缩放及 AMP overflow 轨迹。
这些是本次可训练范围干预的一部分，不能把结果解释成纯粹的 prototype 表示能力证书。

## 内部末 epoch 指标

这些指标来自训练时的转换标签验证范围；smoke/196 张内部验证指标与原始 COCO 全 5,000 张独立评价范围不同。

```json
{
  "metrics/precision(B)": 0.7317463804799142,
  "metrics/recall(B)": 0.677877587247957,
  "metrics/mAP50(B)": 0.7569283156729928,
  "metrics/mAP50-95(B)": 0.600519874343309,
  "metrics/precision(M)": 0.717394899038879,
  "metrics/recall(M)": 0.6557312271413048,
  "metrics/mAP50(M)": 0.7089432593137311,
  "metrics/mAP50-95(M)": 0.48825251784775064
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

SOURCE.json 记录的执行版本：`D:\coco_wire\experiments\acd_proto_tail_unfreeze_20261006\runs\RUN_PROTO_TAIL_ACD_FEASIBILITY_S0\source\train_acd.py`；记录时间=2026-10-06T22:19:00.712914+08:00（Asia/Shanghai）。

| 快照 | SOURCE.json SHA256 | 本地快照 SHA256 核验 |
|---|---|---|
| [实际执行脚本](source/train_acd.py) | `d5868a1d2dc3e57dded535561aec96554fab42f9b0b1ea71c1d863d3e009e763` | 通过 |
| [运行前协议](source/PROTOCOL.md) | `42059829e3e49175a093bb03d2df07e87ac2eb8adc398395ac0fbb22f351be0d` | 通过 |

此处核验 Run/source 快照；项目 scripts 中当前文件的后续编辑不替代历史实际执行版本。
