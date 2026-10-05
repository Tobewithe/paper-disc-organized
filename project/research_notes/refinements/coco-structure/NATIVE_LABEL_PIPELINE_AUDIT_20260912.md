# S020 原生标签诊断有限审计

日期：2026-09-12。审计者：fresh GPT-5.6-Sol ultra，同家族，provisional。总体 **WARN**，未报告fatal问题。

子审计独立读取脚本及产物，核验COMPLETE的492哈希和EXPORT的525哈希；887目标与S019集合相同，5,322行恰为887×6，各分组均值及75计数可重算一致。GT来源、指标归一化、结果存在性、原生overlap的annotation ID回映与几何隔离均PASS。

实际训练走stock SegmentationTrainer→YOLODataset/Format，ExactCOCODataset未被导入实例化；loss包装不替换dataset。243个实际val item重放一致、3,298普通ID匹配；12图实际loader图像与缓存不同已隔离。审计者核对DECODE_WITNESS记录，未自行重新执行所有GPU优化或解码。

WARN范围：选定243图887失败目标、同图GT辅助系数求解、无重新分配、无网络训练或增强训练复现、不是AP。1,363/5,322达到迭代上限，不能称容量上界。新颖性方面mask_ratio/overlap/polygon栅格/高网格proto上采样均为官方现有功能，upstream loss与安装归档SHA一致。本实验不能支持方法创新、历史训练错误或密集专属机制。

证据：`experiments/coco_clean_20260911/native_label_pipeline_probe.py`、`analyze_native_labels.py`，`diagnostics/native_label887_20260912/{protocol.json,metrics.csv,LABEL_ANALYSIS.json,DECODE_WITNESS.json,COMPLETE.json,EXPORT_SHA256.json}`，以及 `diagnostics/native_label_source_20260912/upstream_loss_v8.4.143.py:522`。
