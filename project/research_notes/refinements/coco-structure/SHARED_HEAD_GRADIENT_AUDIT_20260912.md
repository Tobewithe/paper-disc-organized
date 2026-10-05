# S022 有限完整性审查

2026-09-12。Overall WARN：A/B/C/D PASS、E WARN、F real_gt＋fixed-assignment first-order diagnostic。复用GPT-5.6-Sol ultra审查者，same-family/reused-reviewer/provisional，非fresh独立复现。

子代理独立核对含bias的G=g h̄ᵀ代数、同尺度五个互斥分区及符号；887目标无分区遗漏或非有限值，自项偏离1最多1.22e-15，有限差分误差最大3.47e-9。与S020同887目标/243图；前向及raw coefficient误差0、最后线性重建最大4.77e-6。491份COMPLETE和499份EXPORT哈希均匹配、执行脚本SHA一致；复算主反向计数高39/215、其余130/672。

解释限制：adjacent指原始COCO bbox相交，非mask接触；仅计算共享同一尺度cv4最后层参数的正样本。单图mask-only、冻结特征/分配、普通梯度一阶方向，不能推断历史多图训练、MuSGD/Adam、多任务/上游动态、方法AP或训练因果。审查者未重新计算2,000次bootstrap，仅核输入、公式、不变量、主要计数和哈希。

证据：`experiments/coco_clean_20260911/shared_head_gradient_probe.py`以及`diagnostics/shared_head_gradient887_20260912/{protocol.json,ANALYSIS.json,GRADIENT_ANALYSIS.json,targets.csv,witness.csv,COMPLETE.json,EXPORT_SHA256.json}`。
