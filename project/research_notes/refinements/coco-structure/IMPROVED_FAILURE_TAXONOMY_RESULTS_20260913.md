# COCO 失败分类 v3：观测状态与证据层分离

> **2026-09-13 更正，以下旧v5解释停止引用。** 严格失败池为4,260，原4,417中157个存在任何保留的同类Mask75候选；仅排除最佳框候选对应的25个不能得到“无好候选”的4,392池。v4的18,234成功数没有被v5纠正，旧“多计2,238成功”描述不成立。v5的`evidence_tier`仍比较过时状态名，9,980个框差/支持不足状态被落入默认成功提示。后续以[五维状态与AP复算报告](../../experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md)及`diagnostics/coco_failure_dimensions_20260913/instances_improved.csv`为准；历史CSV保留溯源。

对 S048 固定槽位实例表进行一次确定性复算。改版目的不是增加更多机制名称，而是避免把阈值定义或错误表面误写成因果原因。

## 互斥主状态

主状态按“最终槽位 → Box IoU → 框对 GT 支持 → Mask IoU”顺序定义，覆盖 COCO val2017 普通 GT 36,335 个实例：

| 主状态 | GT | 含义 |
|---|---:|---|
| `no_final_slot` | 3,704 | 没有同类最终槽位；上游原因未解析 |
| `box_state_failed_mask_bad` | 6,584 | 有槽位但 Box IoU < 0.75，且固定槽位 Mask IoU < 0.75 |
| `box_state_failed_mask_good` | 1,106 | Box IoU < 0.75，但固定槽位 Mask IoU ≥ 0.75 |
| `box_support_limited_mask_bad` | 1,158 | Box IoU ≥ 0.75、框覆盖自身 GT < 0.95，且 Mask IoU < 0.75 |
| `box_support_limited_mask_good` | 1,132 | 框支持不足，但固定槽位 Mask IoU ≥ 0.75 |
| `candidate_assignment_candidate` | 25 | 当前固定槽位 Mask IoU<0.75，但同图另一同类槽位的最佳框槽位 Mask IoU≥0.75；提示候选身份/保留问题 |
| `mask_readout_failed` | **4,392** | Box IoU ≥ 0.75、覆盖 ≥ 0.95，且没有另一同类高质量框槽位，但 Mask IoU < 0.75 |
| `mask_success` | 18,234 | Box IoU ≥ 0.75、框覆盖 ≥ 0.95 且 Mask IoU ≥ 0.75 |

这些类别是测量状态，不是原因。尤其 `no_final_slot` 不能拆成候选缺失、分类错误、NMS 或分数过滤。进一步检查同图候选后，25 个实例存在“另一同类框槽位本身达到 Mask75”的候选身份线索；其余 4,392 个才是更干净的掩码读出追查池。框差/支持不足但掩码好的实例单独保留，不计入 `mask_success`。这个候选线索仍是诊断证据，不等于自动可用的替换方法。

## 错误表面改为证据标签

对每个实例保留自身遗漏、同类邻居误报、异类邻居误报和背景误报的连续面积比。只有当至少两个表面各占有效 GT 面积 5% 以上时才标记 `mixed_substantial`，其余只标记单一主要表面；因此不会把数值噪声或一个像素误报变成机制结论。在 4,392 个 `mask_readout_failed` 中，错误表面分布与原 4,417 池近似，连续比例仍保留在逐实例表中，论文统计应优先使用连续量。

## 阈值敏感性

中心定义为 Box IoU≥0.75、GT 支持≥0.95、Mask IoU<0.75，并排除 25 个候选身份线索后，掩码读出失败池为 4,392（12.09%）。未排除前为 4,417（12.16%）；支持阈值 0.90/0.98 时的原始池为 5,142/3,320，说明中心切点仍应做敏感性报告。

## 推荐论文用法

正文报告八个主状态和连续错误表面；把 `evidence_tier` 用作措辞约束：`mask_readout_failed` 可称“掩码读出候选失败”，`candidate_assignment_candidate` 可称“候选身份线索”，`box_support_limited_mask_bad` 只能称“支持受限候选”，`no_final_slot` 称“上游未解析”。不要把 `background_fp`、`mixed_substantial` 或 E(4) 分箱直接写成根因。高低拥挤组应在这些主状态内分别报告，避免把目标组成差异误当拥挤效应。

产物：[逐实例表](../../experiments/coco_clean_20260911/diagnostics/improved_failure_taxonomy_20260913_v5/instances_improved.csv)、[汇总及敏感性](../../experiments/coco_clean_20260911/diagnostics/improved_failure_taxonomy_20260913_v5/SUMMARY.json)、[脚本](../../experiments/coco_clean_20260911/improve_failure_taxonomy.py)。
