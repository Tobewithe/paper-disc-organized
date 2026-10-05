# S021 有限完整性审查

2026-09-12。Overall **WARN**。审查者为先前标签审查的GPT-5.6-Sol ultra子代理；新fresh reviewer因thread limit无法创建，因此为same-family、reused-reviewer、provisional，不称全新独立复现。

A GT来源PASS，B归一化PASS，C结果存在性WARN，D执行链PASS，E范围WARN，F类型real_gt＋same-image GT oracle。未发现伪造、标签错配或会推翻诊断的代码错误。

独立核验：887目标243图，6,209=887×7评价行，1,774=887×2校准行；249个COMPLETE SHA无不匹配，exit0。原生640 Format身份/LetterBox和固定640²×native area目标归一化均正确。正尺度/仿射alpha=exp(rho)>0保留像素排序，阈值oracle保留框外GT于分母并处理ties。

数值复核引用执行者见证：同缩放顺序最终损失最大差4.77e-6，全图顺序2.23e-5；FP64 decode仅3/887改变最多1像素，norm restore仅2/887改变最多3像素。未由审查者独立重做全部GPU优化或bootstrap。

限制：校准拟合c·upsample(P)，最终阈值用于upsample(c·P)，浮点差逐目标保存；审查后已在本地分析汇总添加logit_interpolation_max_error和free_interpolation_max_error。LBFGS停止前达不到迭代上限不等于收敛；校准/自由解维数、参数化、精度不同。严格限定同图固定失败队列，不能外推AP、新图泛化、历史训练因果、头容量不足或方法创新。

证据：`experiments/coco_clean_20260911/readout_calibration_probe.py`、`no_candidate_readout_probe.py`及`diagnostics/readout_calibration887_v2_20260912/{COMPLETE.json,ANALYSIS.json,CALIBRATION_ANALYSIS.json,metrics.csv,ranking.csv,calibration.csv}`。
