# 来源与实际阅读范围

1.1新增[评价与决策](references/evaluation-and-decision.md)，从既有评价系统、数学核验和实证综合保存候选身份、有限机会、成本敏感决策与统计反例。来源路径、SHA和选读范围见[整理来源](CURATION_SOURCES.json)。本次没有重算指标或执行新模型实验。

本Skill于2026-11-05从本项目当日已有的TIDE／PyTorch学习记录整理而来，没有新增外部检索或模型实验。项目中的原阅读记录位于`project/research_notes/literature/pig-diagnosis-methods_2026-11-05/README.md`；下列原始来源使Skill可独立携带。

- Daniel Bolya、Sean Foley、James Hays、Judy Hoffman，TIDE，ECCV 2020，[正式DOI](https://doi.org/10.1007/978-3-030-58580-8_33)、[作者PDF](https://dbolya.github.io/tide/paper.pdf)。已读前12页，包含方法、部分实验和尺度分析；附录未系统阅读。PDF SHA256：`e3fe96d7378800c2cc900bdb54f7ef120a6ac70690e0fd82faf23ea3a72cc6ff`。
- [TIDE官方实现固定commit](https://github.com/dbolya/tide/tree/49a5d2a4aeb56795e93a3ed7cc7e6d25757bb4c1)。已核README、错误定义、Data、AP、匹配、`evaluate`与`evaluate_range`相关部分。`tidecv/quantify.py` SHA256：`4234e3cbb26ec22a057d473463e75b0fb36f31e9e6bfdca6d190c44113c9e533`。
- PyTorch 2.5官方[Profiler](https://pytorch.org/docs/2.5/profiler.html)、[CUDA异步执行](https://pytorch.org/docs/2.5/notes/cuda.html#asynchronous-execution)、[Benchmark](https://pytorch.org/docs/2.5/benchmark_utils.html)、[峰值分配显存](https://pytorch.org/docs/2.5/generated/torch.cuda.max_memory_allocated.html)。已读CPU／CUDA活动、预热与同步、测量开销、self／total时间及显存口径；原记录核对了项目Torch 2.5.1的适用API。

Hoiem等2012年相关工作的原文尚未独立阅读全文，未将其列作本Skill已直接掌握的方法来源。TIDE作者结果与文档定义、当前对象的实际验证分别判断。
