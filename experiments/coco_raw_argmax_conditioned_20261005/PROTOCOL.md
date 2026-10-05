# 按 argmax 正确条件重分类 COCO raw 掩码失败

## 问题

复用 `coco_raw_capability_5000_20260916` 的 5,000 张 val2017、36,335 个普通 GT 和完整 raw NPZ。对每个 GT，只保留预测类别 argmax 与 GT 类别一致的 raw 候选，再检查 Box75、Mask75 及两者是否落在同一 raw。

## 互斥分类规则

按以下优先顺序分类：

1. `Semantic fail`：没有任何有效 argmax-correct raw；
2. `Box fail`：有 argmax-correct raw，但没有 argmax-correct raw 达到 Box75；
3. `Success`：存在同一 argmax-correct raw 同时达到 Box75 和 Mask75；
4. `Box/Mask misaligned`：存在 argmax-correct Box75 和 argmax-correct Mask75，但不存在同一 raw 同时达标；
5. `Mask fail`：存在 argmax-correct Box75，但没有任何 argmax-correct Mask75。

这里把“Mask fail”的“无同 raw Mask75”收紧为“整个 argmax-correct 集合没有 Mask75”；否则会与 misaligned 重叠。原始 raw 的类别分数、框 IoU 和掩码 IoU全部来自同一 raw 身份，类别分数不参与几何选择。

## 数据与限制

- 只做已有 raw NPZ 的派生统计，不重新前向、不训练模型。
- 类别 argmax 使用 80 维 raw score 的最大类，按官方 COCO sorted category id 映射回原始 category_id。
- Box75 为 `box_iou >= 0.75`；Mask75 为已解码 raw 的 `mask_iou >= 0.75`，未解码或 NaN 不能算达标。
- 这是 raw 几何能力诊断，不是最终输出召回，也不是 COCO AP 分解。
