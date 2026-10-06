# ACD native coefficient feasibility experiment

更新于 2026-10-06T20:23:01+08:00（Asia/Shanghai）。

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

## 固定范围与限制

Ultralytics 8.4.100，官方 COCO `yolo26m-seg.pt`；仅训练原生 `cv4` / `one2one_cv4` coefficient heads，
其余网络及所有 BatchNorm 冻结。baseline 与 ACD 从相同官方权重重启，seed=0，batch=2，workers=0，输入 640。
两臂先各 1 epoch smoke，再各固定 3 epoch 正式训练；smoke 不是主 AP 结果。
正式 fit 使用已有 796 张 train2017 cohort，训练内验证使用已有 196 张转换标签验证图像。
最终采用固定第 3 epoch 的 EMA coefficients 覆盖原始官方 FP32 模型，保留官方 FP32 frozen 权重；
不选 best epoch，不在推理时使用 action 或 GT，不修改 native one-to-one 后处理。
独立原始 COCO val2017 全 5,000 张评价：FP32、640 square letterbox、conf=0.001、max_det=300、
官方 mask decoder / 原图 scaling、COCO80→91 mapping、pycocotools。绝对 AP 可能不同于 stock rectangular-batch 验证器。

fit cohort 和全验证集均复用历史研究图像，包含其选择限制；全 5,000 张验证集不是新的 blind test。
这是一 seed、短预算的 coefficient-only feasibility screen，不能替代充分训练的结果，也不能量化训练 seed 的不确定性。
ACD auxiliary 固定 weight=0.05、tau=0.50、min gain=0.01、min coverage=0.80、每图每 branch 最多 12 个候选；
观察 paired 结果后不改变超参数。完整规则见 [PROTOCOL.md](PROTOCOL.md)。

## 原始 COCO 全验证集结果

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

## 配对损伤与修复

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

## Run 记录

| Run | arm / purpose | 执行状态 | 训练/评价 receipt | 回传状态 |
|---|---|---|---|---|
| [RUN_ACD_EVAL_SMOKE_S0](runs/RUN_ACD_EVAL_SMOKE_S0/REPORT.md) | 未知 / Independent eight-image evaluator interface check | completed | 评价 complete / 8 图 | verified |
| [RUN_ACD_FEASIBILITY_S0](runs/RUN_ACD_FEASIBILITY_S0/REPORT.md) | acd / feasibility | completed | 训练 complete | verified |
| [RUN_ACD_NUMERICAL_VERIFY_S0](runs/RUN_ACD_NUMERICAL_VERIFY_S0/REPORT.md) | 未知 / Synthetic objective and actual GradScaler optimizer-audit verification; no COCO or AP result | failed | 未知 | verified |
| [RUN_ACD_NUMERICAL_VERIFY_S0_RETRY1](runs/RUN_ACD_NUMERICAL_VERIFY_S0_RETRY1/REPORT.md) | 未知 / Synthetic objective and actual GradScaler optimizer-audit verification; no COCO or AP result | completed | 未知 | verified |
| [RUN_ACD_PAIRED_EVAL_S0](runs/RUN_ACD_PAIRED_EVAL_S0/REPORT.md) | paired / evaluation | completed | 评价 complete / 5000 图 | verified |
| [RUN_ACD_PIPELINE_S0](runs/RUN_ACD_PIPELINE_S0/REPORT.md) | 未知 / Locked sequential smoke, paired training and full evaluation orchestration | failed | 未知 | verified |
| [RUN_ACD_PIPELINE_S0_RETRY1](runs/RUN_ACD_PIPELINE_S0_RETRY1/REPORT.md) | 未知 / Locked sequential smoke, paired training and full evaluation orchestration | failed | 未知 | verified |
| [RUN_ACD_PIPELINE_S0_RETRY2](runs/RUN_ACD_PIPELINE_S0_RETRY2/REPORT.md) | 未知 / Locked sequential smoke, paired training and full evaluation orchestration | completed | pipeline COMPLETE | verified |
| [RUN_ACD_SMOKE_S0](runs/RUN_ACD_SMOKE_S0/REPORT.md) | acd / smoke | completed | 训练 complete | verified |
| [RUN_ACD_SMOKE_S0_RETRY1](runs/RUN_ACD_SMOKE_S0_RETRY1/REPORT.md) | acd / smoke | completed | 训练 complete | verified |
| [RUN_BASELINE_FEASIBILITY_S0](runs/RUN_BASELINE_FEASIBILITY_S0/REPORT.md) | baseline / feasibility | failed | 未知 | verified |
| [RUN_BASELINE_FEASIBILITY_S0_RETRY1](runs/RUN_BASELINE_FEASIBILITY_S0_RETRY1/REPORT.md) | baseline / feasibility | completed | 训练 complete | verified |
| [RUN_BASELINE_SMOKE_S0](runs/RUN_BASELINE_SMOKE_S0/REPORT.md) | baseline / smoke | failed | 未知 | verified |
| [RUN_BASELINE_SMOKE_S0_RETRY1](runs/RUN_BASELINE_SMOKE_S0_RETRY1/REPORT.md) | baseline / smoke | completed | 训练 complete | verified |
| [RUN_BASELINE_SMOKE_S0_RETRY2](runs/RUN_BASELINE_SMOKE_S0_RETRY2/REPORT.md) | baseline / smoke | completed | 训练 complete | verified |

首次 detached SSH 尝试的 `RUN_BASELINE_SMOKE_S0` 保留为失败记录。该尝试在 TRAINING_SETUP 前终止，
没有训练指标；精确终止原因与退出码未观测。首次成功的 baseline smoke 是独立重试 Run `RUN_BASELINE_SMOKE_S0_RETRY1`；
审计修正后本次配对流程采用的 baseline smoke：`RUN_BASELINE_SMOKE_S0_RETRY2`。
上述失败事实以对应 run.json 的 failure_observation 为来源，报告不把重试成功回填为首次尝试成功。

未作为科学比较臂的失败正式训练 Run：`RUN_BASELINE_FEASIBILITY_S0`。
失败审计及已生成的 checkpoint/部分产物继续保留；工程重试保持原 AMP/超参数与固定预算，不回填旧 Run 为成功。

## 核验与门槛判定

独立汇总核验：通过；[READOUT_VERIFICATION.json](runs/RUN_ACD_PAIRED_EVAL_S0/READOUT_VERIFICATION.json)。
该核验从 COCO accumulated precision 重算 AP，并从逐图/逐 detection 记录重算配对计数与点估计；
不重新执行模型、GT 匹配或 bootstrap 区间计算。

- Mask AP：ACD − baseline >= +0.003：未通过；证据 `runs/RUN_ACD_PAIRED_EVAL_S0/SUMMARY.json`。

机器可读判定：[DECISION.json](runs/RUN_ACD_PAIRED_EVAL_S0/DECISION.json)；
其中保留逐项三值判定、所选 Run、数值阈值、原始证据相对路径和 SHA256。
生成脚本：[build_report.py](scripts/build_report.py)。报告生成不会修改任何 Run metadata。
