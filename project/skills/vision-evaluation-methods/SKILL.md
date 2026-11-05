---
name: vision-evaluation-methods
description: >-
  Interpret detection and instance-segmentation evaluation, oracle opportunities,
  conditional action costs, intervention evidence and measured model efficiency.
  Use for TIDE analysis, raw/GT/output comparisons, risk decisions and synchronized benchmarks.
  视觉评价、风险决策、反例与实际成本测量。
metadata:
  version: "1.1.0"
---

# 视觉模型的错误解释与成本测量

本Skill保存视觉评价、风险决策与实际成本知识。TIDE及PyTorch方法见[错误与成本测量](references/methods.md)；raw／GT／输出身份、有限oracle、成本敏感决策与统计反例见[评价与决策](references/evaluation-and-decision.md)，实际阅读范围见[来源说明](SOURCES.md)。

## 根据要回答的问题选择测量

| 问题 | 采用的观察 | 判断范围 |
|---|---|---|
| AP变化来自哪些错误表现？ | 固定原始预测，分别作错误oracle修正，比较分数、覆盖与几何 | 诊断错误影响；训练原因需额外依据 |
| 框定位与Mask质量分别怎样变化？ | BOX与MASK诊断分开，明确IoU阈值和配对对象 | 两种几何和匹配口径分别解释 |
| 哪段计算真正昂贵？ | Profiler记录CPU／CUDA活动及命名范围 | 定位主要开销，注意嵌套与并发 |
| 简化后实际快了多少？ | 独立于Profiler的固定条件预热、同步、重复基准 | 报告端到端中位时延与波动 |
| 显存减少了多少？ | 固定测量区间，分别报告allocated与reserved | 口径一致后比较，峰值和事件总量分开 |

先沿用原任务评价，解释工具与原评价的输入和匹配差异。错误类型、时延与显存回答不同问题，不能互相替代。依据实际测得的主要开销决定是否继续结构修改。

这是已有阅读内容的知识封装，尚未据此完成当前猪模型的TIDE诊断或A6000成本基准；不因Skill存在就填入模型效果。
