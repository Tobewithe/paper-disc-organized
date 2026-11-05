---
name: model-method-selection
description: >-
  Choose vision research methods from adaptation, fusion, losses, auxiliary learning
  and knowledge retention; reason about evidence, research fixation and failure feedback.
  方法选择、科研推理、失败反馈与证据边界。
metadata:
  version: "2.1.0"
---

# 从学习问题选择方法

本Skill帮助形成有文献依据的方法选择。先解释一个做法为什么可能解决当前问题，再给出组合；用户已说明问题时直接提出有理由的建议，无须反复索取完整实验配置。

讨论研究固着、反思、反馈和失败后的更新时，读取[科研推理与反例](references/research-reasoning.md)。它保存已读研究和项目反例，不要求固定表单或强制学习循环。现有Skill只提供已积累知识，不能替代问题驱动的外部学习。

## 思考顺序

1. **说明缺的是什么。** 新模块应带来更多细节、长程关系、对象条件、域不变信息，还是更好的训练信号？把实际观察与解释分开。例如“辅助任务学会但Mask没提高”仍可能是目标不相关、特征未被利用或普通适配损伤。
2. **选择方法族。** 按下表选择相关Skill，并通过宿主的技能发现能力读取全文。通常用1–3项即可，不依赖固定路径、专用命令或工作台。
3. **比较真正不同的选择。** 至少解释一个简单方案与一个有明确额外价值的替代。例如可学习投影与特征蒸馏解决的约束不同，不能只按复杂程度排序。
4. **给出一套连贯的配方。** 说明方法作用位置、提供的先验／监督、与原目标的关系、保留的自由度及可能代价。只为当前缺口添加部件，避免把所有热门方法叠在一起。
5. **交代依据。** 区分论文中定义和观察到的结果、由机制可推导的判断、迁移到当前模型的假说。“经典做法”表示已有明确方法与研究实践，不表示对任意任务最优。

| 当前学习问题 | 读取的Skill |
|---|---|
| 两路特征的统计、基底、语义或域不同 | `feature-adaptation-methods` |
| 应加法、拼接、门控、跨尺度还是注意力融合 | `feature-fusion-methods` |
| 应选BCE、Focal、Dice、Tversky、Lovász或边界目标 | `segmentation-loss-design` |
| 多个目标学习速度不同、互相牵制或优先级不同 | `multi-task-optimization` |
| 新模块训练破坏原有能力，或需要迁移教师知识 | `pretrained-transfer-methods` |
| 需要把上述选择放回YOLOSeg的实例分割机制 | `yoloseg-method-design` |
| 错误、风险决策、评价与实际成本 | `vision-evaluation-methods` |
| 遮挡监督、视频记忆与跨帧身份 | `occlusion-video-identity` |
| 合成分布、几何标签、资产与渲染 | `synthetic-scene-methods` |
| RGB定位、部位owner与模型输出的证据身份 | `visual-grounding-and-evidence` |

## 要交付的方法建议

按任务需要用连贯文字说明：**问题解释 → 首选方法及理由 → 一个有意义的替代 → 推荐组合 → 最可能失败的条件 → 文献依据与关键未知**。可使用公式和示意，但不要把答复变成代码教程、实现检查表或强制实验审批。

评估建议围绕方法区别设计。例如有无特征匹配可区分“适配器增加容量”与“教师表征约束”的作用；同容量普通融合可帮助解释注意力模块的增量。这里的比较建议服务于判断，不要求每个构思先冻结一整套实验。

## 示例：给YOLOSeg拼接一个预训练分支

先判断分支信息与原特征是互补还是应该等价。互补表示可优先考虑投影后的拼接／残差融合；若确实希望学生获取可靠教师的表示，再选择特征、注意图或关系蒸馏。只有数据域发生变化且域不变性合理时，才进一步考虑域对齐。

如果目标是细边界，而输入与读出都已丢失相应分辨率，单换边界loss不会提供新的观测；如果问题是不同实例被当作同一前景，提高语义分割精度也未必解决归属。[YOLACT]、[CONDINST]和[POINTREND]提供理解这一区别的具体架构实例。

损失缩放、梯度冲突和知识保持应分别选择，不由“多个loss”自动推出PCGrad，也不由“特征分布不同”自动推出CORAL。

详细示例见[方法选择案例](references/selection-cases.md)，依据范围见[论文卡片](references/papers.md)及[SOURCES.md](SOURCES.md)。

[CONDINST]: https://arxiv.org/abs/2003.05664v4
[POINTREND]: https://arxiv.org/abs/1912.08193v2
[YOLACT]: https://arxiv.org/abs/1904.02689v2
