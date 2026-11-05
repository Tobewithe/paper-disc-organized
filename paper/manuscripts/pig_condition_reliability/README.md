# 猪实例分割的条件可靠性

本线研究成像、场景和输入条件变化时，哪些能力需要保持、哪些信息需要适应。它连接[跨条件问题](../../../project/outlines/OUTLINE_PIG_GENERALIZATION.md)、[覆盖迁移Idea](../../../project/ideas/IDEA_PIG_COVERAGE_TRANSFER.md)与[受控合成Idea](../../../project/ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)。

## 当前取舍

[成像稳定性V1](../../../project/ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)在固定模型、几何和标签的有限变化中，未达到预定的大幅粗粒度实例成功数损失，当前不据此启动保持训练。结果限定已见TRAIN、低输出阈值和IoU50口径；细边界、排序、新来源及更强条件变化继续开放。

具体条件、真实与合成的分别比较及效应判断见[完整报告](../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)。后续研究需要贴近目标条件的实际失效证据，广义可靠性和受控合成的价值仍按各自Idea判断。

执行记录及协议由[Study](../../../experiments/pig_appearance_stability_20261103/study.json)连接，当前行动见[当前安排](../../../project/ACTIVE_PLAN.md)。[整理前全文](../../../project/research_notes/restart/entrypoints_20261105/LINE_pig_condition_reliability.md)保留首试完整入口与交付记录。稳定身份为`LINE_PIG_CONDITION_RELIABILITY_20261103`。
