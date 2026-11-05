---
name: synthetic-scene-methods
description: >-
  Understand synthetic vision data through conditional scene distributions,
  rendering and label semantics, camera geometry, articulated surfaces and contact.
  Use when generating or evaluating synthetic instances, parts, occlusion and appearance.
  合成数据、几何标签、场景采样、表面与接触知识。
metadata:
  version: "1.0.0"
---

# 合成场景与监督

从目标任务需要的可观察条件、标签含义和资产能力出发。读取[分布、成像与标签](references/distributions-and-labels.md)和[资产、工具与工程反例](references/geometry-and-tools.md)，实际来源范围见[SOURCES.md](SOURCES.md)。

| 问题 | 需要使用的知识 |
|---|---|
| 多生成是否足够 | 联合支持、拒收后的分布、有效几何与独立来源 |
| 图像更真实是否更有用 | 内容／外观差异、标签一致与同预算下游效用 |
| 部位与遮挡真值如何定义 | owner、源表面、V/T/S/P层级及可读性 |
| 单RGB能约束哪些生成参数 | 投影尺度歧义、材质／照明多解、可观察区间 |
| 资产或工具能做什么 | 变形／UV／标签身份、碰撞和支撑的不同资格 |

定义明确、真实RGB可观察、资产可控制、输出可验证是四件事。参数已设置或JSON合法，不代表最终图像满足语义。知识中的旧实验结果用于限定方法，不自动恢复任何生成、训练或已撤销的对抗分支。
