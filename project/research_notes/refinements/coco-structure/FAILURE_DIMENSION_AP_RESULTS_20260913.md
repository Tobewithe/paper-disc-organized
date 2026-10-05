# S067—S069：COCO失败分类、AP机会与矩形范围诊断

当前完整报告及逐实例数据：[REPORT.md](../../experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/REPORT.md)。

重点结果：官方E4低高R75差8.609点中，7.449点在固定槽位框差掩码差状态；类别×大小共同支持标准化后该状态仍贡献6.959点。其理想完整掩码替换AP+5.629，但仅剪去真实矩形外像素AP+1.659，说明粗范围只解释一部分，不能把联合失败直接命名为框错误根因。4,417框好支持充分失败组理想掩码修复AP+4.630，但差距仅缩小0.453点。

![失败机会](../../experiments/coco_clean_20260911/diagnostics/failure_dimension_analysis_20260913/failure_opportunity.png)
