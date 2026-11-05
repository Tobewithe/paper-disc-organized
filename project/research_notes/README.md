# 知识入口与研究笔记

**全部知识统一以Skill存储和维护。** 理论、成熟思想、具体方法、经验、反例，以及阅读和实验形成的认识都在此范围。研究Idea和纲要保留研究组织身份，不纳入知识迁移。完整主份及维护方式见[Skill目录](../skills/README.md)，当前行动见[当前安排](../ACTIVE_PLAN.md)。

## 已保存的知识

| 要理解的问题 | Skill主份 | 主要覆盖 |
|---|---|---|
| 怎样从问题选择方法、从失败更新判断？ | [model-method-selection](../skills/model-method-selection/SKILL.md) | 方法比较、科研推理、固着与反思研究、项目反例。 |
| 不同表示怎样适应和迁移？ | [feature-adaptation-methods](../skills/feature-adaptation-methods/SKILL.md) | 投影、归一化、蒸馏、域相容、原型分配与对应不变性。 |
| 互补信息怎样结合？ | [feature-fusion-methods](../skills/feature-fusion-methods/SKILL.md) | 多尺度、门控、条件读出、共同坐标、梯度与功能依赖。 |
| 分割目标与标签怎样定义？ | [segmentation-loss-design](../skills/segmentation-loss-design/SKILL.md) | 概率/区域/边界/拓扑目标，unknown、owner、支持、归约及风险代理。 |
| 辅助任务怎样共同学习？ | [multi-task-optimization](../skills/multi-task-optimization/SKILL.md) | 权重、速度、梯度、多目标折衷与主任务效用；[辅助学习正文](../skills/multi-task-optimization/references/auxiliary-transfer.md)。 |
| 适配怎样保留已有能力？ | [pretrained-transfer-methods](../skills/pretrained-transfer-methods/SKILL.md) | 参数/函数保持、BN与EMA状态、实际梯度路径、实例蒸馏和实证取舍。 |
| 对象知识怎样进入实例分割？ | [yoloseg-method-design](../skills/yoloseg-method-design/SKILL.md) | 成熟架构、部位迁移、[对象表示数学](../skills/yoloseg-method-design/references/object-representation-math.md)、[读出与项目经验](../skills/yoloseg-method-design/references/readout-and-project-lessons.md)。 |
| 怎样解释评价、机会、风险和成本？ | [vision-evaluation-methods](../skills/vision-evaluation-methods/SKILL.md) | TIDE、raw/GT/输出口径、有限oracle、成本敏感决策、干预反例和实际基准。 |
| 遮挡、完整形状与视频身份怎样关联？ | [occlusion-video-identity](../skills/occlusion-video-identity/SKILL.md) | 遮挡/裁切、amodal监督、因果记忆、重现、数据字段、跟踪评价及延迟。 |
| 合成图怎样形成有效监督？ | [synthetic-scene-methods](../skills/synthetic-scene-methods/SKILL.md) | 联合分布、成像/频谱、几何标签、源表面、资产能力、蒙皮与接触反例。 |
| 模型输出怎样成为视觉证据？ | [visual-grounding-and-evidence](../skills/visual-grounding-and-evidence/SKILL.md) | 定位与实例发现、部位owner、局部观察、坐标、缺失证据及专用接口。 |

每项的`SOURCES.md`、论文卡片和`CURATION_SOURCES.json`保存适用范围与来源层级。文献作者自报、数学推导、本项目实测和设计假说分别解释；Skill存在、读取过、理解掌握和当前效用也分别判断。

## 覆盖与仍有的缺口

[知识覆盖说明](../skills/KNOWLEDGE_COVERAGE.md)及[逐来源清单](../skills/KNOWLEDGE_COVERAGE.json)记录本轮525份本地材料的去向、明确提炼来源和保留角色。历史实验、原阅读与写作材料继续作为时点证据；它们的文件分类不是525份全文阅读或零遗漏证明。

本轮补齐已查出的较早知识主题，未读候选、摘要线索和未验证解释保留原状态。经典多任务共享归纳偏置、任务相关性和泛化条件仍需系统学习。当前猪模型的TIDE诊断、实际成本及新辅助方法验证仍按研究范围待办。

## 获取新知识的工具

[arxiv-search](../skills/arxiv-search/SKILL.md)用于定向检索与PDF处理；[scholar-megasearch](../skills/scholar-megasearch/SKILL.md)用于多来源检索、合并与引用扩展。服务与依赖按执行时核验。

从目标、现象和条件出发，借教材、专著、可靠综述、经典研究及引用脉络学习。已有Skill提供线索，覆盖不足时继续补充；不要求每篇阅读新建一个Skill，也不以当前实验获益作为保存知识的条件。

## 研究组织与原始证据

- [猪实例分割中心纲要](../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)、[学习问题](../outlines/OUTLINE_PIG_TRAINING.md)与[猪体知识Idea](../ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)：研究目标、候选主张与判断，保持原位。
- [辅助学习阅读记录](literature/pig-auxiliary-transfer_2026-11-05/README.md)、[TIDE与成本阅读记录](literature/pig-diagnosis-methods_2026-11-05/README.md)：当时来源、读取范围与分析。
- [近期实验综合](PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)、[基线身份](PIG_YOLO_BASELINE_LINEAGE_20261104.md)与[训练审计](PIG_TRAINING_INTEGRATION_AUDIT_20261104.md)：原条件与时点结论。
- [Idea演变](IDEA_EVOLUTION.md)与[安排历史](restart/ACTIVE_PLAN_HISTORY_20261105.md)：研究过程。

后续知识更新进入Skill，原实验事实与研究安排回到所属记录，笔记通过链接引用知识。详细分工见[工作台指南](../WORKBENCH_GUIDE.md)。
