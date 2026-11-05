---
name: pretrained-transfer-methods
description: >-
  Choose theory-backed strategies to adapt pretrained vision models while retaining useful knowledge: head-first fine-tuning, residual adapters, L2-SP, output/feature distillation and low-rank updates. Use when new modules or auxiliary training damage a strong YOLOSeg baseline. 迁移学习与知识保持方法。
metadata:
  version: "2.1.0"
---

# 预训练迁移与知识保持

先说明要保留哪种能力、需要改变哪种能力。保持参数、保持表示、保持输出、冻结基础函数是不同约束。方法定义与来源边界见[方法说明](references/methods.md)；BN、梯度路由、EMA及实例蒸馏经验见[训练状态与迁移](references/training-state-and-transfer.md)。

## 选择要限制的对象

| 希望限制的变化 | 候选方法 | 主要取舍 |
|---|---|---|
| 强特征初期被随机新头带偏 | 先训练头，再联合微调；差异化适配强度 | 需要初始特征足够有用；新头可能不能单独解决任务 |
| 新任务只需小幅表示增量 | 残差适配器、部分微调 | 限制自由度可帮助保持，也可能限制必要变化 |
| 不希望偏离预训练参数 | L2-SP | 参数接近不等于函数或实例输出接近 |
| 希望保留已有函数响应 | 输出蒸馏、Learning without Forgetting | 教师错误与输入覆盖影响保持内容 |
| 想继承中间结构 | 特征／注意／关系蒸馏 | 不同约束保留不同内容，可能抹平有益差异 |
| 有理由限制更新子空间 | 低秩适配等参数高效方法 | 低秩假设和应用位置需要依据，原LoRA以语言模型为主 |

[LPFT] [ADAPTERS] [L2SP] [LWF] [KD] [FITNETS] [AT] [RKD] [LORA]

## 根据问题构造组合

**新模块随机、基线很强：** 保留基础路径，用残差或适配器学习增量；先让新头／接口适应，再考虑联合微调。LP-FT提供头初始化与特征扭曲的研究依据，但原理论和主要实证来自分类迁移，不是YOLOSeg通用收敛定理。[LPFT]

**目标数据少且与预训练相关：** L2-SP以起点为参数先验，是比单纯向零收缩不同的选择。若目标域需要明显改变表示，过强的起点约束可能妨碍学习。[L2SP]

**需要避免已有输出大幅变化：** 用旧模型的相关响应作训练约束，同时保留任务监督。只保持旧错误不会带来新能力；只在新域输入上保持，也不保证旧域覆盖。[LWF]

**新增能力来自教师：** 按需要选择输出、特征、注意图或关系，不把所有蒸馏项都同时加上。教师应在目标所需能力上可信，参数更多不等于在当前任务更好。

## YOLOSeg里要想清楚的事

对原型—系数掩码，保留每个原型通道可能比保留最终Mask函数更苛刻：基底变化和系数变化可以相互补偿。同一类猪的语义概率蒸馏也不能直接保留实例身份。若比较掩码函数，需要先定义对应的是哪一个对象／空间区域；这是学习目标的定义，不是一个可忽略的实现细节。[YOLACT] [MASKRCNN]

保留原参数文件并使用适配器，不等于适配器启用时的预测永不损伤。小学习率、冻结与蒸馏都是有偏选择，需说明它们怎样在稳定性与可塑性间取舍。

形成建议时给出：保留对象、可塑部分、迁移信号、训练阶段关系及最可能的负迁移；不要以“防遗忘”名称替代目标定义。

来源见[论文卡片](references/papers.md)与[SOURCES.md](SOURCES.md)。

[ADAPTERS]: https://arxiv.org/abs/1705.08045v5
[AT]: https://arxiv.org/abs/1612.03928v3
[FITNETS]: https://arxiv.org/abs/1412.6550v4
[KD]: https://arxiv.org/abs/1503.02531v1
[L2SP]: https://arxiv.org/abs/1802.01483v2
[LORA]: https://arxiv.org/abs/2106.09685v2
[LPFT]: https://arxiv.org/abs/2202.10054v1
[LWF]: https://arxiv.org/abs/1606.09282v3
[MASKRCNN]: https://arxiv.org/abs/1703.06870v3
[RKD]: https://arxiv.org/abs/1904.05068v2
[YOLACT]: https://arxiv.org/abs/1904.02689v2
