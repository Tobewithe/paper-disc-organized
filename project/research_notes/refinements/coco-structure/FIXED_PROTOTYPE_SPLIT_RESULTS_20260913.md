# S074：固定原型的系数读出能力与留出像素检验

日期：2026-09-13。状态：COMPLETE，GT 辅助诊断，无网络训练、无 COCO AP。

## 问题

S070 将裁切与未裁切响应分开后，仍有一批目标在真实解码中失败。S074 检验一个更窄的问题：在原型 `P` 和预测框完全固定时，当前系数是否还有可迁移的修正空间。每个实例使用 640 输入网格中自身 GT 像素和非自身像素，按类别均衡后确定性地分成拟合集与留出集。系数残差只在拟合集上求解，留出集不参与优化。

这不是部署方法。GT 只用于形成诊断标签；模型、框、类别、候选分数和裁切规则没有改变。输入网格 IoU 和留出 AUC 不能当作 COCO Mask AP。

## 协议与完整性

- 队列来自 S070 冻结的 144 个张量：高/低密度各有 `mask_good_control`、`requires_target_pixel_recovery`、`sufficient_true_pixels_but_residual_false_pixels` 三个先验分层，每层 24 个。4 个实例有效正/负支持不足，实际完成 140 个，未按 S074 结果补选。
- 每个实例的原型为 32×160×160，先按官方双线性插值到 640×640；用原预测框得到支持域，排除 crowd。
- 拟合的是 32 维系数残差，目标为 BCE + `0.01` 的 logit 残差平方项 + `0.0001` 的参数 L2，严格凸 Newton/回溯最多 24 步。原始系数、原型和框都保持不变。
- `split_iou` 在完整 640 支持域上解码，但残差只看拟合集像素；`allfit_iou` 是使用拟合集和留出集全部采样点的条件机会对照。`split_hold_auc` 只在未参与拟合的像素上计算。
- 运行目录：`experiments/coco_clean_20260911/diagnostics/fixed_prototype_split_20260913_v5`；原始实例表为 `instances.csv`，协议为 `protocol.json`，脚本为 `experiments/coco_clean_20260911/probe_fixed_prototype_split.py`。

## 原始结果

均值为实例均值，IoU/AUC 在 0–1 量纲；增益为百分点。括号为 5,000 次实例重采样的 95% 区间，未做多重比较校正。

| 密度 | 先验分层 | n | 原始 IoU | split-fit IoU | all-fit IoU | split 增益 | all-fit 增益 | 留出 AUC 增益 | split 达到 IoU≥.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| high | mask-good control | 23 | .7923 | .8529 | .8540 | +6.06 [4.43, 7.71] | +6.18 [4.53, 7.85] | +2.52 [1.18, 3.99] | 22 |
| high | target recovery required | 24 | .4167 | .5808 | .5890 | +16.40 [12.13, 21.30] | +17.23 [12.87, 22.20] | +11.97 [8.19, 16.20] | 4 |
| high | residual false pixels | 24 | .5009 | .7436 | .7571 | +24.28 [18.94, 29.42] | +25.62 [20.24, 30.92] | +16.16 [11.02, 21.61] | 11 |
| low | mask-good control | 23 | .8299 | .8769 | .8781 | +4.70 [3.14, 6.46] | +4.82 [3.26, 6.60] | +1.58 [1.03, 2.19] | 23 |
| low | target recovery required | 24 | .4274 | .6175 | .6309 | +19.01 [12.71, 25.96] | +20.34 [14.07, 27.26] | +14.95 [8.82, 22.09] | 4 |
| low | residual false pixels | 22 | .5143 | .6832 | .6929 | +16.89 [13.04, 20.99] | +17.86 [13.62, 22.43] | +8.61 [4.97, 13.14] | 11 |

## What this establishes

1. On this fixed, outcome-stratified queue, a coefficient correction learned from one set of pixels improves the ranking on disjoint pixels. The effect is not only a threshold or area rescaling: held-out AUC also increases.
2. The conditional opportunity is much larger in the two mask-failure strata than in the mask-good controls. For high density, the split IoU gains are 16.40 and 24.28 points versus 6.06; for low density they are 19.01 and 16.89 versus 4.70. This is evidence that the frozen decoder often has an instance-response readout error where the mask is already failing.
3. The same pattern appears in low and high groups. This experiment therefore supports a general coefficient/readout limitation on the selected failure queue, not yet a density-specific causal mechanism.
4. The original prototype is not shown to be sufficient for every target: full-mask residuals can extrapolate outside the sampled pixels, and all-fit IoU remains below .75 for many cases. Conversely, the held-out gains mean that “the prototype has no usable information” is too strong for this queue.

## Limits and next decision

The queue is a fixed subset of explored validation images and was selected from S070 diagnostic strata. The coefficient solver sees GT and is not a shared predictor; no number here is a method/AP result. The control gains also show that the solver can improve already-good masks, so the comparison is not a proof of dense-only specificity.

S074 makes the next method question precise: can a shared predictor infer the same *response correction* from information available at inference, while preserving the original support and avoiding the S026/S041 failures? A valid next experiment must use a frozen, disjoint image split, train the correction from GT only on the training images, and compare against same-capacity direct pixel supervision and the unchanged coefficient head. It must report full COCO AP, high/low R75, and own/neighbor/background error changes. Repeating GT coefficient fitting or transplanting a source point would add no new evidence.

