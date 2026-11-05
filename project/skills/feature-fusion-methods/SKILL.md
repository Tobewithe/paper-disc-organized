---
name: feature-fusion-methods
description: >-
  Choose principled feature fusion for vision modules: addition, concatenation, residual/gated fusion, FPN/BiFPN, SE/CBAM, non-local attention, FiLM and deformable sampling. Use for combining complementary branches or selecting where and how to fuse in YOLOSeg. 模块融合的方法与理论。
metadata:
  version: "2.1.0"
---

# 特征融合方法

先说明两路信息为什么互补，以及需要在哪个尺度、位置和对象条件下相互作用。融合的价值来自信息与归纳偏置，不来自“又加了一个注意力模块”。详细原理、公式和限制见[方法说明](references/methods.md)；共同坐标、门控梯度、功能依赖和简化经验见[坐标与功能依赖](references/coordinates-and-dependence.md)。

## 方法选择

| 需要的作用 | 候选 | 主要取舍 |
|---|---|---|
| 两路语义相近，简单合并 | 投影后加法 | 假设可在共同表示空间叠加，计算简洁 |
| 希望保留不同特征供下游选择 | 拼接＋学习变换 | 保留来源自由度，增加容量和计算 |
| 强基线之上学习增量 | 残差融合／残差适配器 | 容易表达“不改动”，但不保证训练后保持 |
| 信息可靠性随图像／对象变化 | 条件门控、FiLM | 需要可用且有辨别力的条件 |
| 需要粗语义与细位置协作 | FPN式侧向融合、BiFPN式双向加权 | 重复融合可能冗余；高分辨率有成本 |
| 需要通道或空间重标定 | SE、CBAM | 重加权已有信息，不能替代实例条件 |
| 需要远距离内容检索／相互作用 | 非局部／注意力融合 | 强表达与更高成本，存在跨实例混合风险 |
| 需要自适应空间支持 | 可变形卷积／采样 | 改变取样位置，不等于自动建立正确对应 |

## 组合原则

**先适应再决定交互。** 两路表示基底差异大时，可用独立投影或适当归一化建立消费接口；互补表示不必被拉成相同特征。

**保留强基线时，残差比完全替换更自然。** ResNet的残差学习与Non-local的零初始化输出投影提供相关机制。[RESNET] [NONLOCAL] 这支持以小增量开始的设计，不提供训练后永不退化的保证。

**权重变化的粒度要匹配问题。** BiFPN学习的融合权重通常是全局参数；SE随图像变化的通道权重、CBAM的空间权重与对象条件门控具有不同能力。[BIFPN] [SE] [CBAM] 若两头同类猪需要不同选择，仅有整图通道权重未必足够。

**匹配语义与位置的需要。** FPN将高层语义传向细尺度；Non-local聚合相关位置；可变形卷积学习取样支持。[FPN] [NONLOCAL] [DCN] 它们改变的归纳偏置不同，不宜因为都叫“增强特征”就相互替代。

## 场景建议

- **细边界＋全局对象理解：** 保留较细局部信息，融合粗对象条件；参考PointRend关于精细特征与粗预测互补性的设计。[POINTREND]
- **外部分支有时有帮助、有时误导：** 可考虑门控残差；说明门控由什么信息判断可靠性。小门值也可能意味着任务不需要该分支，不能把门控存在当知识已被利用。
- **远距离可见片段需要归属：** 相关性聚合应受实例条件约束；单纯增强同类别相似性可能把邻猪一起混入。

形成建议时给出一种简单融合、升级理由及可能损伤。若讨论注意力权重，记住其数值受前后层尺度影响，不能直接当独立的因果重要性。

来源范围见[论文卡片](references/papers.md)和[SOURCES.md](SOURCES.md)。

[BIFPN]: https://arxiv.org/abs/1911.09070v7
[CBAM]: https://arxiv.org/abs/1807.06521v2
[DCN]: https://arxiv.org/abs/1703.06211v3
[FPN]: https://arxiv.org/abs/1612.03144v2
[NONLOCAL]: https://arxiv.org/abs/1711.07971v3
[POINTREND]: https://arxiv.org/abs/1912.08193v2
[RESNET]: https://arxiv.org/abs/1512.03385v1
[SE]: https://arxiv.org/abs/1709.01507v4
