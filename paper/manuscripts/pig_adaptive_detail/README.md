# 为困难实例分配视觉细节与计算

本线研究何时局部重看、更多细节或额外计算能够提供有效实例证据，由[按需细节Idea](../../../project/ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)连接[观察问题](../../../project/outlines/OUTLINE_PIG_OBSERVATION.md)。当前受测N为多阶段训练后的项目保留权重。

## 当前取舍与证据

- [局部重看V1](../../../project/ideas/IDEA_PIG_LOCAL_RELOOK_V1.md)：原病例池未满足预定支持量，入口结束，局部推理方法未测。见[资格报告](../../../experiments/pig_local_relook_20261103/REPORT_V1.md)。
- [独立夜间四臂训练](../../../project/ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)：保持项相对普通续训和无保持具有局部作用，但候选均未通过净效用与损伤要求，当前配置关闭。见[训练报告](../../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)。

两项检验回答不同问题。广义细节方向暂缓、判断未知；独立原图信息、局部输入的实际增量和普通适配退化仍值得按证据区分。新观察及其他利用方式保留开放性。

实际执行状态与原协议见[夜间Study](../../../experiments/pig_detail_training_night_20261103/study.json)，当前行动见[当前安排](../../../project/ACTIVE_PLAN.md)。[整理前全文](../../../project/research_notes/restart/entrypoints_20261105/LINE_pig_adaptive_detail.md)保留原数值、反馈及交付入口。稳定身份为`LINE_PIG_ADAPTIVE_DETAIL_20261103`。
