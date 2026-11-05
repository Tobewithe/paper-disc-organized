---
name: multi-task-optimization
description: >-
  Choose main-task-driven auxiliary learning (Auto-Lambda, AuxiLearn, ForkMerge, TAG) and multi-task optimization: fixed weighting, uncertainty, GradNorm, PCGrad, MGDA and CAGrad. Use when auxiliary supervision helps optimization but not target generalization, or when choosing task weights and sharing. 辅助任务效用、泛化与多目标优化。
metadata:
  version: "2.1.0"
---

# 多任务与辅助目标优化

多个loss共存，至少有四类不同问题：**尺度与单位、学习速度、梯度方向、任务价值与优先级**。先分清问题，再选择方法。原理、公式与理论边界见[方法说明](references/methods.md)。当问题是“辅助监督是否真正帮助主任务、为什么训练更好而泛化更差”时，读取[主任务效用与辅助迁移方法](references/auxiliary-transfer.md)，比较TAG、Auto-Lambda、AuxiLearn与ForkMerge。

## 从目标到方法

| 主要需求 | 合理候选 | 不会自动解决什么 |
|---|---|---|
| 简单可解释的任务权衡 | 固定加权的强基线 | 需要依据选择权重与目标归一化 |
| 不同目标似然尺度／噪声 | 同方差不确定性加权 | 不等于每像素可信度，也不判断任务价值 |
| 不同学习速度和梯度幅度 | GradNorm | 不直接处理梯度方向，不保护主任务优先级 |
| 局部梯度相互冲突 | PCGrad | 不保证每步或最终AP都提高 |
| 希望折衷多个同等重要目标 | MGDA／Pareto观点 | Pareto驻点不一定是所需部署折衷 |
| 平均目标优化与冲突兼顾 | CAGrad | 其平均目标不天然等于主分割优先 |
| 任务需要不同表示 | 调整共享／专用表示、分阶段学习 | 单纯缩放loss无法创造兼容表示 |
| 辅助价值随任务／训练阶段变化 | 主任务meta-loss驱动的Auto-Lambda | 一步代理及原SGD式实现需要适配实际优化器 |
| 辅助效用依赖样本或空间交互 | AuxiLearn非线性损失组合 | 额外模型、曲率近似及合法对应的要求 |
| 单步梯度代理不可靠 | ForkMerge有限窗口分叉／验证合并 | 增加训练与选择成本；主任务分支仍可能自身退化 |
| 需要选择任务共享组合 | TAG方向性任务亲和性 | 训练同批亲和性与跨域泛化分别检验 |

[UNCERTAINTY] [GRADNORM] [PCGRAD] [MGDA] [CAGRAD] [MTLSURVEY]

## 主任务与辅助任务要分清

YOLOSeg的框、分类、Mask以及新增部位／边界目标，可能既共享信息又相互牵制。辅助部位准确率并非必然值得与Mask AP同等优化。先明确目标：提升部署分割，还是同时交付多个输出？只有前者时，平等推进所有任务不一定合适。

对辅助任务提出信息理由：它能补哪类监督或归纳偏置，是否学到部署任务不需要的域特征？若任务本身不相关，梯度平衡不能把它变得有价值。

## 选择顺序

1. 将每项loss解释成一个目标，明确其监督对象与统计单位，避免把所有标量随意合并后谈“任务平衡”。
2. 从固定加权开始建立清楚的候选。需要比较动态方法时，说明为何存在尺度、速度或方向问题。
3. GradNorm适合相对学习速度不同；PCGrad适合局部冲突假说；不要把“一个loss数值大”当它主导共享表征的证明。负梯度内积也可能提供有益正则，先比较目标数据上的作用，再决定是否消除。
4. 明确优化对象和主任务优先级。需要选择辅助价值时，比较主任务单独训练、固定辅助与主任务效用驱动的方法；保持相同的基本训练修正。权重或模型合并使用的meta数据属于训练选择过程，独立评价另留。
5. 讨论收益和代价：主任务、辅助能力、计算以及对不同样本的影响，不能只汇报总loss更快下降。真实语义对非语义辅助的增量，与两者相对无辅助的效用分别判断。

## YOLOSeg例子

新增部位loss下降而实例Mask变差：若主任务权重被有效覆盖／归一化改变，可先研究目标加权；若共享表示上长期存在有害交互，可比较冲突处理；若部位与实例归属之间没有有效学习联系，优先重想辅助目标或表示共享。三个解释需要不同方法。

GradNorm、PCGrad和不确定性加权可组合，但组合增加自由度与额外假设。只有它们分别解决已说明的缺口时才推荐叠加。

引用与阅读范围见[论文卡片](references/papers.md)和[SOURCES.md](SOURCES.md)。

[CAGRAD]: https://arxiv.org/abs/2110.14048v2
[GRADNORM]: https://arxiv.org/abs/1711.02257v4
[MGDA]: https://arxiv.org/abs/1810.04650v2
[MTLSURVEY]: https://arxiv.org/abs/2004.13379v3
[PCGRAD]: https://arxiv.org/abs/2001.06782v4
[UNCERTAINTY]: https://arxiv.org/abs/1705.07115v3
