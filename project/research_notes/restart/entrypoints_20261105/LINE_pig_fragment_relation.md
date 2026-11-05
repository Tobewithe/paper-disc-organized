# 入口整理前快照（2026-11-05）

原文件：[paper/manuscripts/pig_fragment_relation/README.md](../../../../paper/manuscripts/pig_fragment_relation/README.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
# 真实猪场的片段归属与实例关系

**研究命题：**当一头猪的可见证据分散时，怎样保持同体归属并区分相邻身体；哪些关系已包含在原独立Mask中，哪些需要新观察或额外标注？

本线来自[实例组织问题](../../../outlines/OUTLINE_PIG_ORGANIZATION.md)与[监督信息价值](../../../outlines/OUTLINE_PIG_SUPERVISION.md)，关联[片段表示Idea](../../../ideas/IDEA_PIG_INSTANCE_INFORMATION_USE.md)。现有论文和方法作为近邻，不将片段归组本身宣称为创新；训练范式、模型和输入尚未预定。

## 当前执行与结果

用户2026-11-03要求整理三层纲要并依据Idea开主线做实验。本线已完成[真实TRAIN适用性首试](../../../ideas/IDEA_PIG_FRAGMENT_RELATION_PILOT.md)，所属Study为`STUDY_PIG_FRAGMENT_RELATION_S0_20261103`，冻结协议见[PROTOCOL_S0.md](../../../../experiments/pig_fragment_relation_20261103/PROTOCOL_S0.md)。

**当前缓存入口已结束，主线暂缓方法训练。** Slurm1157完整测量1272图／14642实例，仅14实例／14图提供可靠原断连支持，未达预定128／32。剔除重叠后增加150个断连不能冒充物理可见真值，故不将它们改作主对象。完整判断与追溯见[REPORT_S0.md](../../../../experiments/pig_fragment_relation_20261103/REPORT_S0.md)。

本次投入30 CPU-core秒、0 GPU秒、0优化步，16项产物全SHA核验回传，12张固定预览逐张查看。这个结论只限制直接使用当前160网格缓存训练显式断片的方案，没有检验关系模型、人工成本或正常效用。更清楚的可见标注或真实归属错误证据可以提供继续依据；目前没有已具备条件的训练后继。

## 结果如何回到知识

每个实际Run对应当时Idea、协议、输入及源码SHA。结果更新具体Idea，回到[片段归属缺口](../../../outlines/OUTLINE_PIG_MODEL.md)、[错误分母缺口](../../../outlines/OUTLINE_PIG_TASK.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)。记录排除范围、未知与重开依据；不为失败强行追加方法。

本线规则见RESEARCH_RULES.md（原路径：`paper/manuscripts/pig_fragment_relation/RESEARCH_RULES.md`；[服务器归档回执](../../maintenance/nonrun_file_compliance_20261104T093520Z/ARCHIVE_RECEIPT.json)）。执行总控见[ACTIVE_PLAN.md](../../../ACTIVE_PLAN.md)，SPM和其他线的历史冻结条件各自保持。
