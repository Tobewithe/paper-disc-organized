# 按 argmax 正确条件重分类 COCO raw 掩码失败

## 结论

这套分类可以回答一个更直接的问题：

> **在 raw 候选已经有正确类别 argmax 后，实例是否仍因框或 mask formation 失败？**

复用 `coco_raw_capability_5000_20260916` 的完整 raw NPZ，对 COCO val2017 全部 5,000 张图片、36,335 个普通 GT 重算。每个 GT 只保留 raw 预测类别 argmax 与 GT 类别一致的候选，并在同一 raw 上检查 Box75、Mask75。

结果是：**语义 argmax 不是主要瓶颈；在语义正确后，mask 形成失败是最大的单一失败类。**

| 状态 | GT数 | 占全部GT | 判定 |
|---|---:|---:|---|
| Semantic fail | 44 | 0.12% | 没有任何 argmax-correct raw |
| Box fail | 6,335 | 17.43% | 有 argmax-correct raw，但没有 argmax-correct Box75 |
| Mask fail | **6,459** | **17.78%** | 有 argmax-correct Box75，但整个 argmax-correct 集合没有 Mask75 |
| Box/Mask misaligned | 484 | 1.33% | argmax-correct 集合中有好框和好 mask，但没有同 raw 双达标 |
| Success | **23,013** | **63.34%** | 同一 argmax-correct raw 达到 Box75 和 Mask75 |
| 合计 | 36,335 | 100% | 五类互斥且穷尽 |

在有 argmax-correct raw 的 36,291 个 GT 中，Mask fail 仍占 17.81%，Box/Mask misaligned 占 1.33%。因此，条件化后最明确的对象是：**类别 argmax 正确、框达到 Box75、但任何同类正确 raw 都没有 Mask75 的 6,459 个实例。**

## 按目标大小

| 尺度 | GT数 | Semantic fail | Box fail | Mask fail | Misaligned | Success |
|---|---:|---:|---:|---:|---:|---:|
| small | 15,264 | 27 (0.18%) | 5,213 (34.15%) | **3,888 (25.47%)** | 282 (1.85%) | 5,854 (38.35%) |
| medium | 12,392 | 11 (0.09%) | 877 (7.08%) | 1,742 (14.06%) | 155 (1.25%) | 9,607 (77.52%) |
| large | 8,679 | 6 (0.07%) | 245 (2.82%) | 829 (9.55%) | 47 (0.54%) | 7,552 (87.02%) |

小目标同时有更高的框失败和 mask 失败比例；大中目标在 argmax 正确后主要已经成功。这支持把后续 mask 形成改进优先放在 **Box75 已成立的小目标和中等目标**，而不是把所有失败都归因于分类。

## 与已有分数无关五类的关系

旧 raw 报告在不使用类别分数的完整候选池上得到：

- 同候选 Box75+Mask75：23,805（65.52%）；
- 好框但整个候选池无 Mask75：6,688（18.41%）；
- 好框和好 mask 分属不同候选：531（1.46%）。

新的 argmax 条件统计得到 23,013 个 Success、6,459 个 Mask fail、484 个 Misaligned。两套数字不能直接当作同一分类的替代品：旧表回答“几何候选池是否存在能力”，新表回答“在类别 argmax 正确的候选子集上，几何能力如何”。新表的价值是把类别语义条件固定后再看 mask formation。

## 口径和限制

- argmax 使用每个 raw 的 80 维 score 最大类，按官方 COCO sorted category id 映射到原始 category_id；没有使用最终 conf、top-k、NMS 或 COCO 输出。
- Box75 和 Mask75 始终来自同一个 raw 身份。`Mask fail` 已收紧为“没有任何 argmax-correct Mask75”，这样与 `Box/Mask misaligned` 互斥；若只写“没有同 raw Mask75”，两类会重叠。
- 这是 raw 能力诊断，不是最终输出召回，也不是 COCO AP 的分类条件分解。raw argmax-correct 不等于该候选一定通过最终置信度、top-k 或 NMS。
- 该分析复用了既有 raw cache，没有重新前向或训练模型。

## 对主线的影响

当前最清楚的失败对象可以写成：

> **类别 argmax 正确、预测框达到 Box75，但 prototype 到 mask coefficient 的形成没有产生 Mask75。**

它比原来的“框好、掩码差”更严格，因为先排除了 raw 候选层面的语义错误。后续空间系数场或 prototype→coefficient 改进，应把这 6,459 个 Mask fail 作为主要机制对象，同时保留 484 个 Misaligned 作为候选位置选择的次要对象。

本报告不改变既有 raw 主实验的五类结果，也不把这组 GT 条件统计当作可部署方法效果。
