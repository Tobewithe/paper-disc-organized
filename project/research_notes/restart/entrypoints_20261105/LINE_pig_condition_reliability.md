# 入口整理前快照（2026-11-05）

原文件：[paper/manuscripts/pig_condition_reliability/README.md](../../../../paper/manuscripts/pig_condition_reliability/README.md)。本页保留整理前全文；其中的状态和行动属于原记录时点。当前内容请回原入口。仅调整相对链接位置，恢复信息见[清单](MANIFEST.json)。

<!-- preserved-history-begin -->
# 猪实例分割的条件可靠性

本线研究成像、场景和输入条件变化时，什么能力应该保持、哪些信息需要适应，以及怎样区分可控合成现象与真实任务证据。它来自[跨条件能力问题](../../../outlines/OUTLINE_PIG_GENERALIZATION.md)和[覆盖／场景依赖缺口](../../../outlines/OUTLINE_PIG_TRANSFER_GAP.md)，不会把光照或某个模型预定为全部问题的根因。

当前首试为[成像外观稳定性Idea](../../../ideas/IDEA_PIG_APPEARANCE_STABILITY_V1.md)，连接[受控合成认识工具](../../../ideas/IDEA_PIG_SYNTHETIC_COUNTERFACTUALS.md)与[覆盖迁移](../../../ideas/IDEA_PIG_COVERAGE_TRANSFER.md)。先用固定模型、几何和标签做条件干预，完成真实TRAIN色调变化与现有30组物理照明对照；不新增训练或合成。

执行入口为[Study](../../../../experiments/pig_appearance_stability_20261103/study.json)及[PROTOCOL_V1.md](../../../../experiments/pig_appearance_stability_20261103/PROTOCOL_V1.md)。主张是首个探索验证的条件敏感性，不是正常方法增益、背景捷径识别或跨农场效用。

## 首试已完成及当前判断

[成像敏感性完整报告](../../../../experiments/pig_appearance_stability_20261103/REPORT_V1.md)：128真实TRAIN、1460实例在固定色调操作下最坏成功数损失2／1460＝0.136986%，95%声明group簇区间[0,0.395849%]；30组合成A/B在owner有效域内各成功278，逐对差异0。两侧基线支持充足，但均低于原5%／10%效应线。

据此结束“大幅粗粒度实例成功数损失”的具体入口，当前不启动针对这组变化的保持训练。真实已见TRAIN及IoU50／低输出阈值触顶，限制了对细边界、AP、分数排序和新条件的解释；广义跨条件问题继续开放，需要实际新条件失效依据才决定后续。受控合成作为认识工具的价值不随单个假说阴性被否定。

工程1158、正式1159均完成，合计56GPU秒／224CPU-core秒、476前向、0优化。54项原产物、158个原标签输入核SHA回传；444个IoU矩阵由原标签与完整RLE独立重算误差0，另一种最大流匹配与区间也一致。8张固定预览已实际查看，事后显示阈值说明另列，不改变冻结判定。

具体规则见RESEARCH_RULES.md（原路径：`paper/manuscripts/pig_condition_reliability/RESEARCH_RULES.md`；[服务器归档回执](../../maintenance/nonrun_file_compliance_20261104T093520Z/ARCHIVE_RECEIPT.json)）。每次结果更新具体Idea、有关纲要和本页；工程无效、低基线支持不足与科学阴性分开。原片段入口关闭、旧SPM阴性和暂停矩阵各自保留，不继承其预算或恢复其配置。
