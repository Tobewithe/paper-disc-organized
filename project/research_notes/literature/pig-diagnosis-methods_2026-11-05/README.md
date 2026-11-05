# 未决问题的外部方法学习：错误归因与成本测量（2026-11-05）

**知识保存补充（2026-11-05）：** 本页已读的TIDE与PyTorch测量知识现统一保存于[vision-evaluation-methods](../../../skills/vision-evaluation-methods/SKILL.md)。下方保留原学习过程、来源和当时状态，通用方法正文后续在Skill中维护；这次整理没有新增外部阅读或模型实测。

用户明确要求：内部复盘发现重要能力缺口后主动向外学习，在相关研究中反复使用、发现新问题、继续学习，逐步补足所需能力。本页记录第一轮针对性学习及其对研究行动的影响。来源结合正式出版索引、作者论文与实现、版本化官方技术文档；arXiv为其中一个入口。没有新建Study／Run，也没有运行猪数据评价或模型前向。

当前问题来自[夜间实验报告](../../../../experiments/pig_multi_idea_night_20261105/REPORT_V1.md#feedback-cycle-v1)：A的AP下降尚未分解到候选覆盖、排序与Mask质量；B的功能简化尚未形成真实成本收益。先补这两个分析所需的方法与工具，后续随结果继续学习优化、迁移或结构方法。

## 1. 实际检索与阅读范围

| 来源 | 本轮取得什么 | 阅读或核验范围 |
|---|---|---|
| Crossref／Springer正式出版记录 | [TIDE，ECCV 2020](https://doi.org/10.1007/978-3-030-58580-8_33)，Daniel Bolya、Sean Foley、James Hays、Judy Hoffman | 题名、作者、年份、正式DOI与出版社；元数据核验 |
| 作者公开论文 | [TIDE作者PDF](https://dbolya.github.io/tide/paper.pdf) | 已读前12页，包含方法第2节、误差独立修正、部分实验及尺度分析；附录尚未系统阅读 |
| 作者官方实现 | [TIDE固定版本](https://github.com/dbolya/tide/tree/49a5d2a4aeb56795e93a3ed7cc7e6d25757bb4c1) | README、错误定义、AP和输入Data实现，以及quantify中的匹配、evaluate、evaluate_range和错误汇总相关部分 |
| PyTorch 2.5官方文档 | [Profiler](https://pytorch.org/docs/2.5/profiler.html)、[CUDA异步执行](https://pytorch.org/docs/2.5/notes/cuda.html#asynchronous-execution)、[Benchmark](https://pytorch.org/docs/2.5/benchmark_utils.html)、[峰值分配显存](https://pytorch.org/docs/2.5/generated/torch.cuda.max_memory_allocated.html) | CPU／CUDA活动、测量开销、预热与同步、重复测量、self／total时间和峰值显存口径；核对项目Torch 2.5.1的适用API |
| 基础近邻索引 | [Hoiem等，Diagnosing Error in Object Detectors，ECCV 2012](https://doi.org/10.1007/978-3-642-33712-3_25) | 仅核正式发表信息；本轮对其方法差异的认识来自TIDE的比较，尚未独立阅读全文 |
| arXiv及其HTML入口 | [TIDE 2008.08115v2](https://arxiv.org/abs/2008.08115v2)、[方法正文HTML](https://ar5iv.labs.arxiv.org/html/2008.08115) | 核身份、摘要及方法正文，与上述同一工作对应，未计成另一项独立证据 |

初始arXiv关键词查询出现天文学及另一篇同名TIDE，随后按完整题名、作者和正式DOI消歧。Semantic Scholar接口返回429，ECVA目录返回403；两者未读取的内容没有进入本轮方法依据。作者公开PDF可正常取得，不受这两条访问失败影响。

## 2. 从TIDE学到的错误归因方法

TIDE将预测中的问题分为跨类别误分、几何定位、二者同时错误、重复检测、背景误检与遗漏GT。通过对一种错误作理想修正，计算`ΔAP = 修正后AP − 原AP`，衡量它对性能的影响。每类修正都从同一原始预测出发；逐类累积修正会引入顺序偏置。不同修正有交互，不能把各类ΔAP直接相加当成AP损失的唯一分解。

它比只数错误或只看总AP多提供了两类信息：错误在分数排序中的影响，以及遗漏与误检对AP的相对影响。论文中的重打分案例显示，总AP变化小也可能同时包含某类错误下降和另一类错误上升。当前据此选择TIDE式归因作为A组诊断的一个工具，保留原COCO统计作为效果权威。

官方实现核对带来以下当前任务所需的具体用法：

- `evaluate`默认分析IoU50；`evaluate_range`虽计算多个IoU下的AP，只在选定的`pos_threshold`上运行错误分析。因此应显式设置与GT75问题相应的阈值，或逐阈值分别分析，不把默认dAP50直接解释成AP50:95的误差贡献。
- `BOX`与`MASK`模式的Loc分别由框IoU和Mask IoU定义。必须分开读取两种诊断，避免将Mask几何问题直接称为框定位退化。
- 在预测和GT类别映射一致的单类猪任务中，跨类别误分项可能恒零；这没有覆盖分类分数下降、阈值筛除和排序退化。需同时比较原输出的分数、图内／全局名次及几何覆盖。
- Data要求bbox为`[x,y,width,height]`，Mask为多边形或COCO RLE；默认每图上限100。转换时需保留图像和实例身份、原画布、零检测图、空Mask、ignore及原分数／顺序，并与原评价的maxDets和匹配口径核对。
- Missed GT oracle通过改变有效GT分母分析遗漏的影响，没有实际产生新的检测；相应dAP用于诊断优先级。研究中仍需另外实现并检验能否修复那些错误。

下一步A的有限CPU诊断据此细化为：先核输入与原统计的基础AP口径，再比较5／10／20轮的框、Mask错误类型和分数／覆盖变化，并按尺度、来源和配对seed看哪些变化一致。若TIDE基础AP与原统计有差异，先解释匹配、ignore、截断或输入转换差异，保留原COCO结果。该工具能帮助定位退化表现在哪类错误，训练原因仍需后续针对性观测与方法学习。

## 3. 从官方资料学到的成本测量方法

B下一步分为两个不同的测量任务：

1. 用Profiler的CPU与CUDA活动和命名范围定位原生网络、候选、采样、解码及策略的开销。区分self与包含子调用的total时间，避免重复累计嵌套算子时间；并发CUDA工作还需结合时间线阅读。
2. 用单独的正常推理基准测实际端到端时延。预热后重复测量并记录中位数及波动，固定设备、线程、输入和数据驻留条件。CUDA操作异步，应使用正确同步的墙钟计时，或明确仅测设备工作的CUDA Events；Benchmark Timer提供预热、线程设置与必要同步支持。

PyTorch 2.5官方资料还明确：shape与stack记录有额外开销，shape记录会暂时持有Tensor引用；`with_flops`只估算特定算子，例如矩阵乘法与2D卷积。故定位开销的Profiler采样与最终正常时延基准分开进行，FLOPs完整性单独注明。`max_memory_allocated`记录Tensor峰值分配显存，测量前可重置峰值统计；它与reserved显存、Profiler分配事件之和分别报告。

这使B的下一步从比较整项Slurm耗时，具体化为可定位开销的CPU／CUDA时间线和独立端到端基准。先查目标分支的实际成本，再决定有无值得继续学习和实现的结构压缩；原15分钟单GPU诊断预算保持。

## 4. 学习进度与研究接续

本轮已取得来源、读过上述范围的方法与实现，并据此细化A/B分析。尚未在本项目预测上运行TIDE，也未在A6000上进行成本测量；它们继续受远端SSH阻塞，原训练和统计结论保持。TIDE兼容性及实际基准结果须在对应环境运行后报告。

接续时按[当前安排](../../../ACTIVE_PLAN.md)和原有限预算执行。若诊断指向排序、候选几何、Mask形状、训练适配或数据／计算瓶颈，就沿实际缺口继续查相关方法及近邻工作，选择最有支持的技术并用于下一步。此轮没有为四个候选Skill创建空目录；是否维护Skill由后续调用需要决定。

## 5. 来源定位

作者公开PDF的SHA256为`e3fe96d7378800c2cc900bdb54f7ef120a6ac70690e0fd82faf23ea3a72cc6ff`，5469948字节，阅读时临时提取文本，未作为实验产物留存。TIDE代码固定到commit `49a5d2a4aeb56795e93a3ed7cc7e6d25757bb4c1`（2020-08-31），本轮取得的文件SHA256为：

- `README.md`：`bc70de5fc7b6bfd893698d138620ac1fa70620b6227e183af82bbbb4d0b1e1e2`
- `tidecv/quantify.py`：`4234e3cbb26ec22a057d473463e75b0fb36f31e9e6bfdca6d190c44113c9e533`
- `tidecv/errors/main_errors.py`：`306feaf50fa5b10a7440e9bfe8a6fe8b78b21a9df7a428a6cf282b0c3422f145`
- `tidecv/data.py`：`75fec02808c990904ec73b0ccd0e1ad45884be24665c67ac2a4f1f6397df1e32`

代码版本与上述来源于2026-11-05检索阅读，固定代码信息核验时点为`2026-11-05T11:28:24.043023+00:00`。PyTorch文档按2.5版本引用；研究判断引用实际读到的范围。
