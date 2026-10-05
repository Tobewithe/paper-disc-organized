# COCO 分层失败分类改进

本次改进不是重新跑模型，而是对 S048 固定槽位结果和 S056 关系连接结果做一对一复算。输入为 COCO val2017 的 36,335 个普通 GT；`iscrowd`/ignore 单列，关系拓扑仅作为上下文。

## 分类原则

将五个问题分开：

1. **最终是否有同类槽位**：没有槽位只能叫 `no_final_slot`，不能推断原始候选不存在，也不能区分分类、NMS 和分数过滤。
2. **同一槽位的框质量**：Box IoU 以 0.75 为主阈值。
3. **框是否覆盖自身 GT**：对框好槽位记录 GT 像素支持率；0.95 只作预先规定的诊断阈值。
4. **掩码结果**：固定同一槽位的 Mask IoU 以 0.75 划分成功/失败。
5. **错误表面**：保留自身遗漏、同类邻居、异类邻居、背景四个连续面积比，并另标混合错误。主导表面只是摘要，不是因果标签。

真实边界拥挤使用连续同类 E(4)，并提供 zero、positive-low、positive-high 的描述性分箱；分箱不能替代连续分析，也不定义错误原因。C/I/L/O/M/X 关系拓扑单独连接，不再等同于失败类别。

## 全量互斥结果

| 分层结果 | GT |
|---|---:|
| `mask_success`（固定槽位 Mask IoU≥0.75） | 20,472 |
| `no_final_slot` | 3,704 |
| `box_limited_mask_bad` | 6,584 |
| `box_limited_mask_good` | 1,106 |
| `box_good_support_limited_mask_bad` | 1,158 |
| `box_good_support_limited_mask_good` | 1,132 |
| `box_good_support_sufficient_mask_bad` | **4,417** |
| `box_good_support_sufficient_mask_good` | 18,234 |

这里的 4,417 才是“同一框已较好、框对自身 GT 支持充分、掩码仍失败”的主要掩码读出失败池；1,158 个支持不足者不能直接归因到系数或原型。框好支持充分但掩码成功的 18,234 个是重要正对照。

## 拥挤分箱的描述性检查

E(4) 的正暴露实例中位数为 0.282。按预先定义的 zero、positive-low、positive-high 汇总，`box_good_support_sufficient_mask_bad` 占比分别为：zero 11.91%、positive-low 12.38%、positive-high 12.64%。该单调差异很小，不能单独宣称拥挤造成掩码失败；S056 的固定 GT 同类删除只在高 E(4) 组显著缩小 R75 差距，提供了更直接的任务级定位证据。

## 这套分类相对旧版的改进

- **互斥**：每个 GT 只有一个 `failure_scope`，不会同时落入框失败和掩码失败。
- **支持单列**：把框几何支持不足与真正的掩码质量失败分开。
- **错误面连续化**：保留四个原始面积比和 `mixed`，避免“主导错误”掩盖多种污染。
- **关系降级为上下文**：O/M/X 描述候选连接拓扑，不再被解释成失败机制。
- **拥挤连续化**：保留 E(4) 原值和 undefined，分箱只用于读表，不按 AP 结果反向选阈值。

## 论文可用表述

> 我们将实例失败按最终槽位、框质量、框对 GT 的支持和固定槽位掩码质量逐层分解，并将错误像素去向及候选关系作为独立字段。COCO val2017 中，4,417 个实例同时满足 Box IoU≥0.75、框覆盖自身 GT 至少 95% 且 Mask IoU<0.75，证明存在数量充足的“框与支持均可用、掩码仍失败”现象；该池应作为掩码分支机制分析对象，而不是将所有框好掩码差实例直接归因于系数或原型。

## 限制

这是最终输出的归因诊断，不是官方 AP 的加性分解。`no_final_slot` 没有上游 source 证据时不能细分。支持率和错误表面是操作性量，需结合官方 COCO AP、E(4) 连续值和独立图片验证；不得把 5% `substantial_surface` 标记写成公认标准。

产物：[分层实例表](../../experiments/coco_clean_20260911/diagnostics/layered_failure_taxonomy_20260913_v2/instances_layered.csv)、[汇总](../../experiments/coco_clean_20260911/diagnostics/layered_failure_taxonomy_20260913_v2/SUMMARY.json)、[脚本](../../experiments/coco_clean_20260911/refine_failure_taxonomy.py)。
