---
name: feature-adaptation-methods
description: >-
  Select theory-backed methods for incompatible visual representations: learned projection, normalization, feature/attention/relational distillation, conditioning and domain alignment. Use when combining modules with different feature distributions, semantics or pretraining. 特征适应与表征迁移方法选择。
metadata:
  version: "2.1.0"
---

# 特征适应与表征迁移

“分布不同”不是一个足够具体的学习问题。先区分统计尺度、表示基底、语义粒度、空间对应、数据域以及任务相关性。读[方法原理与选择](references/methods.md)获取公式和主要限制。涉及原型分配、局部对应、不变性或合成迁移时，再读[对应与原型学习](references/correspondence-and-prototypes.md)。

## 从差异选择约束

| 主要差异 | 合理候选 | 关键前提 |
|---|---|---|
| 通道数、幅度、统计尺度不同 | 可学习投影；BN／GN／适当标准化 | 不应删除对目标有用的幅度／风格信息 |
| 两套网络表达同一内容但基底不同 | 投影后特征匹配；注意图或关系蒸馏 | 应匹配的内容和样本／位置关系明确 |
| 两路提供互补信息 | 分别适配后融合；条件调制 | 不必强制它们成为相同表示 |
| 需要随对象／场景改变使用方式 | FiLM式特征调制、门控 | 条件信息可用且与任务有关 |
| 源域与目标域确有分布偏移 | CORAL、DANN等域适应候选 | 可共享的判别信息存在，标签语义相容 |
| 图像位置／物体形变不对应 | 几何对齐、可变形采样或对应学习 | 通道归一化不能解决位置错配 |

## 最有用的区别

**投影是自由度，蒸馏是约束。** 一个由下游任务训练的投影可以学习怎样消费新特征，不需要两路逐值相同。FitNets额外要求投影后的学生表示预测教师提示；这会引入教师偏好及过度约束风险。[FITNETS]

**选择真正需要保留的不变量。** Attention Transfer保留空间激活分布，容许通道差异；RKD保留样本间距离／角度关系，容许更大表示变化。二者都不自动保留逐像素实例身份。[AT] [RKD]

**归一化不等于语义适配。** BN与GN改变统计和优化参数化，不能仅凭均值方差接近推断内容一致。BN“通过消除内部协变量偏移解释全部收益”的原始说法受到后续研究挑战。[BN] [GN] [BNMECHANISM]

**域不变性需要任务相容。** CORAL匹配协方差，DANN让域难以区分；只对齐边缘分布可能混淆条件类别／实例。已有理论反例说明相同的特征分布可以仍然具有相反的标签关系。[CORAL] [DANN] [INVARIANT]

## 推荐组合的形成

1. 说明新分支应贡献什么：边界、形状、部位还是上下文。
2. 先选需要保留的关系，决定“保留差异后融合”或“向可靠参考学习”。
3. 统计不稳定时再选合适归一化；不要把小batch作为无条件替换整个预训练网络归一化的理由。
4. 有教师时，只匹配对目标有意义的内容层级；没有可靠教师时，让任务监督训练适配器是一项合理基线。
5. 说明什么现象会支持该选择、什么反例会让你放弃它；不把更小的对齐距离直接当分割收益。

## YOLOSeg例子

给颈部引入一个预训练分支，若其优势是长程关系，可保留局部CNN分支并融合，未必应该用逐特征MSE抹平差异。若目标是把教师的边界定位能力迁入原网络，可比较有位置对应的特征提示与注意图迁移。若主要问题是源图与真实猪场的标签／可见性语义不同，先解决任务相容，盲目域对齐可能更差。

输出方法选择、适应对象、保留的自由度、配套监督及文献依据。方法迁移到YOLOSeg仍是待验证设计。来源见[论文卡片](references/papers.md)与[SOURCES.md](SOURCES.md)。

[AT]: https://arxiv.org/abs/1612.03928v3
[BN]: https://arxiv.org/abs/1502.03167v3
[BNMECHANISM]: https://arxiv.org/abs/1805.11604v5
[CORAL]: https://arxiv.org/abs/1607.01719v1
[DANN]: https://arxiv.org/abs/1505.07818v4
[FITNETS]: https://arxiv.org/abs/1412.6550v4
[GN]: https://arxiv.org/abs/1803.08494v3
[INVARIANT]: https://arxiv.org/abs/1901.09453v2
[RKD]: https://arxiv.org/abs/1904.05068v2
