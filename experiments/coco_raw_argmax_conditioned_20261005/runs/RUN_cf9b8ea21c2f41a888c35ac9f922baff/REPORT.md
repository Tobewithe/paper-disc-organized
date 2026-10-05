# argmax 正确 raw 候选池的可挽回能力

这份派生分析把两个问题分开：

1. argmax 正确的 raw 候选池中，是否存在 Box75、Mask75 和同候选双达标；
2. 正常输出已经失败的 GT，raw 候选池中是否仍存在可以挽回的候选。

## 1. 全部 GT 的 raw 能力

在 36,335 个普通 GT 中，只看 argmax 与 GT 类别一致的 raw：

| 能力 | GT数 | 占全部GT |
|---|---:|---:|
| 存在 argmax-correct Box75 | 29,956 | 82.44% |
| 存在 argmax-correct Mask75 | 24,053 | 66.20% |
| 存在同一 argmax-correct raw 同时 Box75+Mask75 | 23,013 | 63.34% |

这三个数字分别回答框能力、掩码能力和联合能力，不能把不同 raw 的框和 mask 拼成一次成功。

## 2. 正常 Mask75 失败实例的可挽回空间

正常输出 Mask75 失败共有 13,568 个 GT。对这些 GT，仍然在完整 argmax-correct raw 池中查找：

| 候选池中存在的能力 | GT数 | 占正常失败 |
|---|---:|---:|
| argmax-correct Box75 | **7,855** | **57.89%** |
| argmax-correct Mask75 | 1,594 | 11.75% |
| argmax-correct 同候选 Box75+Mask75 | **1,233** | **9.09%** |

因此，正常输出失败中有 7,855 个实例曾经存在类别正确且框质量足够的 raw 候选；其中 1,233 个甚至存在类别正确、框和掩码都达标的同一 raw。这 1,233 个是最直接的“输出阶段没有保留好候选”对象，适合研究分数、top-k、NMS 和输出竞争。

剩余大部分失败没有 argmax-correct Mask75。它们即使放宽输出筛选，也不能仅靠重新保留现有候选解决，更接近 mask formation 本身的不足。

## 3. 按原始输出失败原因拆分

| 原正常输出状态 | GT数 | argmax Box75 | argmax Mask75 | argmax 联合75 |
|---|---:|---:|---:|---:|
| raw 候选池没有 Mask75 | 11,512 | 6,232 | 0 | 0 |
| 有好 mask，但 GT 类别分数≤0.001 | 1,797 | 1,414 | 1,363 | 1,053 |
| 有好 mask，但未进入 head top-300 | 153 | 113 | 127 | 93 |
| COCO 每类 maxDets=100 排除 | 103 | 93 | 101 | 84 |
| COCO 配对竞争 | 3 | 3 | 3 | 3 |
| 合计失败 | 13,568 | 7,855 | 1,594 | 1,233 |

11,512 个 raw geometry unavailable 的实例没有任何 raw Mask75，因此不能靠输出筛选挽回；1,797+153+103+3 中则有一部分存在 argmax-correct 的好 mask，属于输出保留或竞争层面的可挽回对象。

## 4. 对后续改进的划分

- **Mask formation 主对象**：类别 argmax 正确、Box75 存在，但 raw 池没有同类 Mask75；这部分需要改 prototype→coefficient→mask 形成链路。
- **输出保留对象**：存在 argmax-correct 同候选 Box75+Mask75，但正常输出失败；这部分需要查分数、top-k、NMS 或 maxDets，不能拿来证明 mask 形成能力不足。
- **候选错位对象**：argmax-correct 好框和好 mask 分属不同 raw；数量较小，作为位置选择的次要方向。

这说明后续强干预不能只在 TAL 正样本上看训练损失，也不能只用最终输出统计。训练阶段保持官方 TAL 梯度路由，机制评价使用完整 argmax-correct raw 池，最终效果再用正常 COCO 输出评价。

## 5. 限制

raw 候选池的“存在好候选”是 GT 辅助能力诊断，不是模型实际会选择的结果。它不等于部署召回，也不代表只要调整置信度就能实现该收益。所有统计复用既有 raw NPZ 和 COCOeval 对账结果，没有重新前向或训练。
