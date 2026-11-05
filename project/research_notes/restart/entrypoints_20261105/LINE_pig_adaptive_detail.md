# 入口整理前快照（2026-11-05）

原文件：[paper/manuscripts/pig_adaptive_detail/README.md](../../../../paper/manuscripts/pig_adaptive_detail/README.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
# 为困难实例分配视觉细节与计算

本线`LINE_PIG_ADAPTIVE_DETAIL_20261103`研究何时局部重看、更多细节或其他计算能提供可用的实例证据。首试受测N是多阶段训练后的项目保留YOLO权重，不预定最终架构或方案。

当前夜间训练已完整结案：[R/L/D/P四臂](../../../../experiments/pig_detail_training_night_20261103/REPORT_V1.md)各16500步，1165–1171训练、评价、统计与终态全部COMPLETED。四组Mask AP为52.734108／54.284877／51.143308／55.249199，均低于原生N57.343500；P相对D提高4.105891并降低损伤，但仍未通过原成功损伤及正常效用护栏。具体Idea记结束／混合，广义方向暂缓／未知；停止当前配置，不自动扩训或加seed。最新来源见[RESULT_SUMMARY.json](../../../../experiments/pig_detail_training_night_20261103/RESULT_SUMMARY.json)，原[交接](../../../../experiments/pig_detail_training_night_20261103/HANDOFF.md)保留启动时快照。

前一[局部重看V1](../../../ideas/IDEA_PIG_LOCAL_RELOOK_V1.md)已按原停止出口结束：128TRAIN只提供15个合格失败，不足32，独立复核一致，方法未测，见[报告](../../../../experiments/pig_local_relook_20261103/REPORT_V1.md)。夜间Study是随后新授权的训练问题，没有为保小试而改分母或追加图像。

2026-11-04[综合反馈](../../PIG_RECENT_EXPERIMENT_SYNTHESIS_20261104.md)已写入[具体夜间Idea](../../../ideas/IDEA_PIG_DETAIL_TRAINING_NIGHT_V1.md)：P高于同条件普通R2.515091 AP，仍低于多阶段保留N，不能混同相对训练作用与采用价值。该区别经[广义细节Idea](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)回到[观察问题](../../../outlines/OUTLINE_PIG_OBSERVATION.md)及[中心纲要](../../../outlines/OUTLINE_PIG_INSTANCE_SEGMENTATION.md#tested-ideas)，关闭配置与广义未知保持；没有新增训练。

数据、实现、冻结源、权重与逐例证据全部留登记服务器，本地只保留研究总结及指针。结果分别反馈到[广义Idea](../../../ideas/IDEA_PIG_ADAPTIVE_DETAIL.md)及[细节缺口](../../../outlines/OUTLINE_PIG_DETAIL_EVIDENCE.md)。规则见RESEARCH_RULES.md（原路径：`paper/manuscripts/pig_adaptive_detail/RESEARCH_RULES.md`；[服务器归档回执](../../maintenance/nonrun_file_compliance_20261104T093520Z/ARCHIVE_RECEIPT.json)）。
