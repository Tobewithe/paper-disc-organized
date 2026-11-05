# 当前科研安排

本页只保留当前任务、有效暂停事项和接续入口。研究线身份见[研究线登记](research_lines.json)，工作方式见[工作台指南](WORKBENCH_GUIDE.md)。

## 当前任务（2026-11-05）：主线与Idea入口精简已完成

按用户要求，已精简9个主线入口和17个已有反馈的Idea，保留当前主张、取舍、关键未知和证据入口；数值、运行状态与预算分别引用原报告、Study和协议。另8个简短Idea保持原文。详情与核验见[维护说明](research_notes/maintenance/entrypoint_simplification_20261105/SUMMARY.md)。

[整理前历史](research_notes/restart/entrypoints_20261105/README.md)完整保留原入口、指南、登记与本页旧内容。视频遮挡表征入口及登记依据已有验收报告补正为原试验已完成、配对无预定实用增量；其他Idea的研究状态和证据身份保持。

[知识入口](research_notes/README.md)继续连接13个Skill的完整主份与加载副本，理论和经验由Skill维护；原证据与历史按时点保存。服务器仍不可用，当前研究执行继续按下列范围待恢复。

## 当前研究判断与入口

研究背景为[真实猪场实例分割中心命题](outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md)，既有探索见[SPM主线](../paper/manuscripts/spm_yoloseg/README.md)。广义[猪体知识Idea](ideas/IDEA_PIG_KNOWLEDGE_FOR_INSTANCES.md)仍为unknown。

[夜间三组实验](../experiments/pig_multi_idea_night_20261105/REPORT_V1.md)已结案：A当前语义配置停止，B共享保留图为结构精简优先候选，C当前局部保持配置停止；真实计算收益及新辅助方法效果尚未验证。完整数值与依据由报告承载。

[辅助学习、迁移与实例使用的阅读记录](research_notes/literature/pig-auxiliary-transfer_2026-11-05/README.md)已形成方法与适用性认识，知识由相关Skill主份维护，当前版本见清单。学习成果与当前模型效用分别判断；实际新增训练和梯度验证未运行。

## 待研究恢复后接续

| 事项 | 范围与入口 |
|---|---|
| A已有预测逐例诊断 | 3臂×3seed×3学习点；零训练、零新前向，4CPU不超过30分钟。见[报告第8节](../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)。 |
| B真实成本测量 | Profiler定位与正常同步预热基准分开，1GPU不超过15分钟；按同一报告第8.6节预算与停止条件执行。 |
| 辅助方法有限验证 | 结合A的共同退化诊断，另立执行合同；保留匹配R与C/Q，按实际优化器核对虚拟更新。设计依据见[方法学习记录](research_notes/literature/pig-auxiliary-transfer_2026-11-05/README.md)，不挪用A的零前向预算。 |
| 更新文档远端归档 | 恢复后另存新版并逐SHA核验；此前结案与反馈回执保持对应时点的快照，不覆盖本轮编辑。 |

## 继续有效的暂停与边界

- 原分配—部位—Mask矩阵评价仍暂停，状态见[对应Idea](ideas/IDEA_PIG_ASSIGNMENT_PART_MASK.md)。
- LCF正式统计结案及后继实验仍暂停，保留`evaluated_pending_statistics`、已有验收结论与未完成项，见[所属Study](../experiments/mask_local_connectivity_server_20261020/study.json)。
- test426继续封存，原SPM相对N至少+5 AP目标未达；已关闭配置按原结论保留。
- 本轮未连接服务器，不据旧快照声明实时队列状态；工作台索引与Git提交／推送未执行。

## 历史位置

此前累计安排已移至[截至2026-11-05的历史记录](research_notes/restart/ACTIVE_PLAN_HISTORY_20261105.md)，完整保留原正文与相对链接的目标。查历史时按对应时点理解其中的“当前”，后续任务直接更新本页当前内容，具体过程留在所属报告。
