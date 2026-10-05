# Box-relative mask readout

本稿依据空间掩码读出的已完成诊断与方法实验建立，与历史CCL、P3稿及笔记本RCMC稿保持独立论证。当前为研究写作版，未投稿；三种子、边界评价及结构控制已完成。普通系数MLP接近空间头总体AP，空间头剩余优势集中在小目标Mask75召回；当前先定位这个差异，再决定方法扩展，不宣称已达到一区录用水平。

- `MANUSCRIPT.md`：英文研究稿（只写已获得证据）。
- `EVIDENCE_MAP.md`：主张与原始运行对应。
- `NOVELTY_AND_POSITIONING.md`：已核验的相近文献及创新边界。
- `references.bib`：核验引用。
- `SUPPLEMENT.md`：方法实现、完整指标、各seed、统计口径及失败记录。
- `VENUE.md`：目标期刊与完成标准。
- `SUBMISSION_REVIEW.md`：现有证据、创新风险和待完成判断。
- `REPRODUCIBILITY.md`：执行依赖、原始运行和公开复现尚需补齐的部分。
- `NEXT_DECISION_20260923.md`：本地评价后的取舍与下一项有限诊断，作为当前规划入口。

当前方法主实验：`experiments/coco_template_source_20260921`；机制证据：`experiments/coco_spatial_calibration_mechanism_20260922`；低开销原生读出验证：`experiments/coco_native_spatial_readout_20260922`。实验各自保留原Study/Run身份，论文只引用，不复制实验。
