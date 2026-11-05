---
name: segmentation-loss-design
description: >-
  Choose segmentation objectives from their statistical and geometric biases: BCE/CE, Focal, Dice/IoU, Tversky, Lovasz, boundary and topology losses. Use to design YOLOSeg mask losses or auxiliary supervision and explain precision/recall, instance identity and AP tradeoffs. 分割损失的方法选择。
metadata:
  version: "2.1.0"
---

# 分割损失选择

按实际误差与目标选择loss。先说它改变了什么学习偏好，再说明能否帮助当前实例分割。读[损失原理](references/methods.md)获取公式、变体和反例；未知标签、实例owner、训练支持及归约见[标签支持与风险](references/label-support-and-risk.md)。

## 首先明确对象

- 每个实例的二值Mask，还是一张互斥的语义分类图？独立sigmoid与跨类softmax表达不同目标。[MASKRCNN]
- 目标是像素概率、区域重叠、漏分／误分代价、边界几何，还是满足特定拓扑？
- 错误来自掩码形状、对象归属、漏检／匹配还是分数排序？Mask AP不只取决于已匹配掩码的平均IoU。[MASKSCORING]
- 监督描述可见区域还是完整身体？没有看到不等于应补出；遮挡后可见区域可以不连通。

## 候选与适用偏好

| 候选 | 优先解决的学习问题 | 主要代价／限制 |
|---|---|---|
| BCE／CE | 局部分类／概率拟合 | 大量易像素或类别比例可主导；未直接优化区域比值 |
| Focal | 易样本数量主导学习 | 更强调困难噪声；不是普遍鲁棒损失 |
| Soft Dice／IoU | 对区域重叠更敏感 | 梯度耦合、空目标、平滑与聚合方式影响结果 |
| Tversky | 明确调节FP／FN偏好 | 减漏分可增加邻猪侵入，不自动提高AP |
| Lovász-hinge／Softmax | Jaccard/IoU的排序型代理 | 二值与多类版本不同；不直接等于COCO实例AP |
| 距离型边界loss | 几何边界偏差有实际意义 | 对标签边界噪声、分辨率和距离尺度敏感 |
| clDice等拓扑目标 | 任务确需管状／连通结构保持 | 原假设不普遍适用于被遮挡的动物可见Mask |

## 有理由的组合

**BCE＋区域重叠项**是值得比较的简单组合：局部分类与区域重叠提供不同梯度。它不是所有数据的默认最优；有效区域、实例／像素聚合决定真正权重。

**区域项＋边界项**适合区域大致正确、轮廓误差重要且标签足够可靠的情形。输入和表示若已丢失细节，换loss不会凭空恢复信息。[BOUNDARY] [POINTREND]

**FN代价增强**适合漏分比误分更昂贵的目标，但应同时解释FP，尤其是拥挤同类实例。不要把β较大当通用Tversky配置；不同实现甚至交换α／β定义。[TVERSKY]

## 形成选择的步骤

1. 说明最需要改善的误差，而非从loss名字开始。
2. 选择一个已有简单目标和一个针对该误差的候选，解释梯度偏好差别。
3. 说明计算对象与聚合：每实例、每图、全batch、前景／背景及可见有效区域。
4. 若组合，说明两项为何互补；同一类重叠代理重复相加可能只增加调参自由度。
5. 把形状改善、实例归属、召回与排序分别解释；避免用损失下降或一个区域指标替代完整任务判断。

## YOLOSeg例子

漏掉细腿时，先判断特征分辨率和部位信息是否存在。若可辨但训练偏好不足，可研究BCE＋Dice或受控FN代价；若读出缺少细节，考虑细特征与对象条件。若两头猪被合成一片，单一前景Dice无法表达“应属于哪一个实例”，需要先确定实例条件／对应目标。[YOLACT] [CONDINST]

来源见[论文卡片](references/papers.md)和[SOURCES.md](SOURCES.md)。

[BOUNDARY]: https://arxiv.org/abs/1812.07032v4
[CONDINST]: https://arxiv.org/abs/2003.05664v4
[MASKRCNN]: https://arxiv.org/abs/1703.06870v3
[MASKSCORING]: https://arxiv.org/abs/1903.00241v1
[POINTREND]: https://arxiv.org/abs/1912.08193v2
[TVERSKY]: https://arxiv.org/abs/1706.05721v1
[YOLACT]: https://arxiv.org/abs/1904.02689v2
