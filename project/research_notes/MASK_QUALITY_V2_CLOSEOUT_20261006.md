# 质量排序头 V2 结案记录

日期：2026-10-06（Asia/Shanghai）

## 结论

独立质量头在训练目标上快速收敛，但没有带来可复现的 COCO Mask AP 增益，因此停止把“crop-IoU 质量预测 + 类别分数重排”作为性能提升主线。

V2 只训练 13,953 个质量头参数，冻结原模型参数和 BatchNorm 状态，使用 Ultralytics 8.4.100 固定运行时。用户在第 5 轮中止训练；第 4 轮 checkpoint 是最后一个完整 checkpoint。

质量 BCE：0.40764 → 0.06692 → 0.05980 → 0.05899。

在完整 COCO val2017 5k、同一 `model.val` 评价下：

| 模型 | Box AP50-95 | Mask AP50-95 | Mask AP50 |
|---|---:|---:|---:|
| 官方预训练基线 | 0.51756 | 0.42155 | 0.65559 |
| V2 epoch4，alpha=1.0 | 0.51699 | 0.42117 | 0.65491 |
| V2 epoch4，alpha=0.25 | 0.51747 | 0.42149 | 0.65544 |

因此质量头对局部 crop-IoU 的拟合没有转化为最终候选保留收益。早期训练内存模型的 0.42276 没有在保存 checkpoint 的独立复评中复现，不作为证据。

## 实现审计

本轮修正了两类实现问题：

1. 冻结原模型时同时冻结 BatchNorm running statistics；否则训练会污染原模型特征分布。
2. Ultralytics 的保存 checkpoint 使用 `checkpoint['ema']` 承载实际模型；质量头评估必须先挂载新增模块，再恢复 EMA state dict，不能直接用原始 YAML 加载后随机创建质量头。

此前受这两类问题影响的中间 Run 保留为历史或无效评估，不与 V2 结果混合。

## 路线决策

当前证据支持“质量头能学习局部掩码质量”，但不支持“局部 crop-IoU 是有效的全局候选选择目标”。后续方法应直接针对已有失败类型和最终任务效用设计监督，不再通过延长同一训练、扩大解冻范围或继续扫描排序强度挽救该路线。

可进入下一轮预审的方向是功能空间信赖域读出：使用候选 coefficient 与固定 prototype 响应构造低维可实现的修正，再由固定求解器约束保护区和修正范数。该方向必须先做不训练的 oracle 投影审计，确认能保留 oracle 收益并减少误伤后才训练。

证据入口：

- `experiments/mask_quality_boundary_branch_20261004/PROTOCOL_V2.md`
- `experiments/mask_quality_boundary_branch_20261004/FACTS.md`
- `experiments/mask_quality_boundary_branch_20261004/scripts/train_boundary_quality.py`
- `experiments/mask_quality_boundary_branch_20261004/scripts/eval_standard_val.py`
