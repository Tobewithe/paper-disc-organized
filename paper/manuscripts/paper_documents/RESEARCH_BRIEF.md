# 研究简报：密集猪群实例分割中的实例碎片化与候选竞争

> 建立日期：2026-09-04。本文档将 `C:\Dpan\codexproject\pigcv_research` 的既有分析接入当前 ARIS 项目；原始结果、代码和数据不复制、不改写，仍以原路径为准。

## 问题陈述

密集接触猪群中的实例分割不仅会出现传统的高重叠重复预测，还可能出现同一真实实例被多个低重叠局部预测分别解释的实例碎片化。现有离线失败分析显示，普通 Mask-NMS 能处理大量高重叠重复，但不能直接处理低重叠、互补的局部实例假设，以及完整候选与局部候选之间的错误排序或共同存活。

本研究的主线是一个“诊断到改进”的综合任务：先系统探明密集场景中实例分割失败率高的深层机制，再针对被证实的主导机制设计可实现、可证伪的改进，并检验其是否同时改善整体分割指标与关系型失败。候选竞争、遮挡关系、层/集合结构、中心聚类等都只是待比较的机制族；当前没有选定方法或论文主贡献。

## 背景

- **领域**：计算机视觉、智能养殖、实例分割。
- **子方向**：密集/接触目标实例分割、实例碎片化、候选或 query 竞争、实例关系建模、遮挡结构与评测诊断。
- **文献状态**：Research Wiki 当前有 27 个论文页面；核验状态以各页和 literature registry 为准，检索记录包含重复与待核验条目。Mask Scoring R-CNN、TIDE、Topology-Aware Query Selection 和 QCell 已按官方来源核验；2026-09-07 新增 Faro 数据集论文的元数据/schema 核验，具体遮挡标注政策仍未核验。页面保留项目相关性边界，本轮不构成全面查新通过。
- **已尝试的方法**：对 YOLO26-seg 的固定预测缓存进行了全量错误分类、Mask-NMS 干预、残留错误分析、互补区域分析、box-mask 来源分析、raw candidate 追踪和完整/局部候选竞争分析。
- **已排除方向**：将普通高重叠重复作为主创新；固定缓存中，Mask-NMS@0.80 已将 427/675 个 O 类 GT 转为 C，来源感知抑制未显示相对普通 Mask-NMS 的额外价值。

## 约束条件

- **运行环境**：PowerShell 7.6.4；深度学习使用 Conda `pytorch` 环境。
- **旧项目证据源**：`C:\Dpan\codexproject\pigcv_research\findings.md`、`research-log.md`、`research-state.yaml`、`密集猪群实例分割论文大纲.md` 与 `artifacts\analysis\`。
- **数据与验证目标**：PigLife、FaroPigSeg、BamaPig2D；方法验证必须包含整体 AP、密集场景与关系型失败，至少多数据集，优先加入跨架构验证。
- **目标**：形成有模型贡献、证据可审查的 SCI 二区论文；当前不启动完整 pipeline 或正式训练。
- **跨架构恢复状态（2026-09-07 核对）**：旧“MaskDINO loading mismatch 尚未修复”已过时。20260825 训练对齐 summary 和预测实际存在：426 图、9,342 个预测，missing/unexpected keys 均为 0。旧三模型联合状态与调整统计有归档，但部分原始分析路径失效，且旧 score .50/含弱关联与当前 .05/核心边不同；恢复和核验前不直接合并为当前正式机制证据。

## 期望方向

- [x] 在现有密集实例分割问题上寻找有文献依据的方法改进。
- [x] 保留已完成的诊断性失败分析，作为方法动机和后续机制指标。
- [ ] 在文献与可行性评估前自动选定网络结构、损失函数或论文主贡献。

## 领域知识与已有证据

- 固定 YOLO26-seg 分析覆盖 7,374 个 GT；基线总失败率为 19.84%。
- Mask-NMS 后残留 196 个 O 类 GT，其中描述性标签为 `LOCAL_REPEAT=65`、`SPLIT_TYPE=50`、`MULTI_PRED_COMPLEX=25`、`OTHER_UNRESOLVED=56`。
- `SPLIT_TYPE` 的多预测联合 IoU 增益中位数为 0.084，联合 coverage 增益为 0.143；这些是有限样本上的描述性证据，不能外推为所有密集实例的机制。
- 在 50 个 `SPLIT_TYPE` GT 中，完整 raw 候选缺失 11 个、存在但丢失 8 个、完整和局部候选共同存活 31 个；共同存活组中局部候选相对完整候选的 score gap 中位数为 0.185。
- 50 个 `SPLIT_TYPE` 中 36 个表现出 box 级参与或 box-mask 混合错误；这支持“并非纯 mask 构造问题”的工作假设，但不构成因果证明。

## 非目标

- 不将 `SPLIT_TYPE=50` 当作全部密集实例分割问题。
- 不把 Touching 直接等同于遮挡，或在未核验 GT modal/amodal 语义前把 GT 外区域称为背景错误。
- 不以 Mask-NMS、启发式后处理或少量失败案例的继续细分作为主要论文创新。
- 不把伪标签、模型一致性或未完成的 MaskDINO 结果写为人工 GT 或正式性能结论。

## 下一阶段

继续 ARIS `idea-discovery` 的机制探索：以 YOLO26 为主线，在 PigLife、FaroPigSeg、BamaPig2D 上复核候选缺失、完整/局部候选竞争、遮挡边界、box-mask 混合和实例合并/碎片化；跨架构比较暂缓。先做控制变量与反事实分析，再根据证据比较干预路线。文献结论和诊断结果共同决定后续是否进入 `research-refine` 或 `experiment-plan`；在此之前不创建方法 `research_contract`。

## 2026-09-05 诊断更新

- PigLife 与 FaroPigSeg 的 same-forward full trace 已完成；重放审计覆盖 586 图、8,327 个最终 mask，raw 张量、Top-K、最终 source mapping 及 batch-2 mask 重建逐项一致，XOR 为 0。
- 全部 1,191 个基线失败 GT 的 raw→Top-K→confidence→final mask-IoU 可用性：PigLife 为 390/390/373/373（394 个失败 GT），FaroPigSeg 为 746/698/553/553（797 个失败 GT）。
- 在 mask IoU >= 0.50 条件下，PigLife 有 4/394、FaroPigSeg 有 51/797 个失败 GT 没有单独合格的 raw mask；Faro 另有 48 个在 Top-K、145 个在置信度门首次失去可用候选。此统计尚不能区分框定位、prototype 表达、高精度 mask 质量和标注语义，也不能证明通用排序根因。
- R006 v1 的 ADDITION、REMOVAL、ADDITION_REMOVAL 分别恢复 65、74、799 个失败 GT，联合替换造成 1 个原本正确 GT 退化。这是指定 GT-guided 输出集合操作实际达到的结果，未证明是全局最优上界，也不能解释排序、NMS 或 mask 形成的因果贡献。
- batch-2 final-mask parity 已通过；Terra 已修复 R006 一对一匹配，并修正旧阶段分析的 I/L/S box-purity 预筛选。v2 全量重跑与统计审计完成：四条件全部 6,226 个 GT 标签及 1,136 个已选 source 与 v1 一致；按 source 排序后的 8,327 个最终 mask 也全部 XOR=0。旧 trace 没有记录历史 framework prototype 实参；新 forward 的实参相等与旧 raw mask 重建一致已验证。

联合增删的失败恢复率及 image-cluster 95% CI 为 PigLife 322/394=81.73% [76.84%,86.46%]、Faro 477/797=59.85% [56.34%,63.21%]；原本正确 GT 的退化为 0/4,080 和 1/955。

R005b 严格质量与含正确邻居的 all-GT 匹配已完成并通过独立完整性审查：586 图、6,226 GT、193,413 条边、final-source XOR=0。coverage >= .75 且 purity >= .75 的失败 GT raw 可用性为 PigLife 360/394=91.37%、Faro 490/797=61.48%；final 可用/同时匹配为 312/310 和 224/224。匹配只限制 source 不重复分配，不能代替输出关系恢复。Faro 的严格单 mask 质量缺口较大；当前尚未区分 box、prototype、mask head 或标注语义的影响。

固定 conf=.05 下，所有过阈值 raw source 都已经在 Top-300，故仅扩大 Top-K 对两份 trace 均无效。下一步是从同一基线分别检验 confidence retention、删除对照和联合操作，保留整体 AP、全部 GT 的恢复/回退；不在探索性 public test 上调参。结果见 `research-wiki/yolo26_strict_quality_20260905.md`，顺序见 `refine-logs/EXPERIMENT_PLAN.md`。candidate-scorer/ranking Gate 保持未通过，Terra 完整性审查为 same-family / provisional。

## 2026-09-07 续接结果

R005c v2 已完成五条件全量计算，新增统计位于 `experiments/yolo26_confidence_statistics_20260907_v1/`。固定 Top-300 下统一将 confidence .05 降至 .01，PigLife 恢复 2、退化 136，Faro 恢复 10、退化 133；失败率分别增加 2.995 和 7.021 个百分点。Faro AP 同时提高 0.580854 点，说明 AP 与固定输出关系指标不同向。GT 联合增删恢复全部 43 个所选目标及 2 个邻居，仅占全部失败 3.78%；没有可部署选择规则或两数据集一致 AP 增益。详见 `research-wiki/yolo26_confidence_diagnostic_20260907.md`；独立审计正在完成。

用户已选择下一优先级为 Faro 单 mask 质量：框定位、mask 表达、标注语义。R005d 首项检查已覆盖 160 图、1,752 GT：307 个 raw 严格 mask 缺失失败中，37 个没有任何 raw 框覆盖到 .75 GT，270 个存在足够框支持但仍无合格 mask。原始 YOLO 标签与 COCO 逐图栅格 mask 多重集合全部一致，3,484 个最终 mask 均处于 source 框支持内。独立 OpenCV/COCO 实现已复现全部 7,008 条覆盖计数并核验 489 个哈希。框支持仅为必要条件，不能据此认定 prototype 或 mask-head 缺陷。详见 `research-wiki/faro_mask_quality_20260907.md`；官方元数据和 schema 已核验，但可访问资料未说明 modal/amodal、遮挡和截断政策，期刊/预印本全文获取受限，因此语义继续未核验。

当前仍在既有 run 的 `idea-discovery` 诊断阶段，candidate-scorer/ranking Gate 保持未通过。下一步应沿上述 37/270 固定分层核对标注政策并审查同 source 裁剪反事实协议；不自动选择方法、训练或进入论文写作。

## 2026-09-07 全量外部推理完成

R012 已完成 FaroPigSeg 全部 train/val/test 和 BamaPig2D 全部 train/eval：4,858 图、27,745 个 GT、45,664 个预测。固定 PigLife 微调 checkpoint，Faro 全量 mask AP 为 39.81、非 C 失败率为 46.83%；Bama 为 64.15、23.34%。Faro 三个划分 AP 均约 39.9，性能缺口在扩大数据范围后仍然存在，但尚不能据此判断具体原因。

旧 492 图的预测列表逐项一致，2,900 个 GT 分类无变化，6,397 个输入/源码哈希通过核验。Bama 存在 77 对文件内容完全相同的图像，其中 11 对跨 train/eval；77 对均有程度不同的标注差异，10 对实例数不同。按授权保留全部记录，后续独立分组评估需要处理这一事实。现有基线总覆盖量含 PigLife 为 5,284 图、32,219 GT，raw 候选完整诊断仍为旧 586 图范围。

结果见 `research-wiki/yolo26_external_full_20260907.md`。用户要求继续后，R012b 已完成 Bama 重复数据敏感性评测：按每组最小/最大图像编号保留一份时，AP 均约 64.45；剔除全部重复组为 64.75，较原全量上升约 0.30/0.60 点。规则预先固定，未按模型表现选择标注；评测样本变化不是模型提升。当前优先利用扩充后的现有数据推进已选 Faro 质量诊断；新增第四数据集仍是条件性选项，方法贡献、因果归因和改进效果仍未成立。

## 2026-09-07 O 判别标准审查

用户明确当前优先关注失败分类与机制，并要求审查 O 判别标准。当前全量实现以 `IoU >= .50 OR GT coverage >= .50` 建核心图，再把单 GT / 多预测连通分量记为 O；由于 IoU 不超过 coverage，这实际上是 coverage .50 的强关联标准。审查复现全部 27,745 个 GT 标签，发现不等分碎片存在结构性漏纳：真实缓存中 Faro 245 个、Bama 29 个当前 MISS，可由多个 coverage < .50 的高纯度片段联合达到 coverage/purity .75。这些只是几何筛查阳性，尚不是人工确认的新 O 标签。

现有 O 中包含 prediction-pair IoU >= .80 的比例为 Faro 53.48%、Bama 86.92%，因此 O 不能直接当作纯碎片化数量。统一给所有边加 purity .50 又会破坏 M/X 关系识别。结论是保留现有强关系基线，讨论独立的局部片段关联、重复/互补/混合细分类及跨 GT 复核；本轮未更改规则、阈值或方法 Gate。详见 `research-wiki/o_criterion_review_20260907.md`。

## 2026-09-07 共同机制目标审查

按用户给出的十项风险完成项目证据审查，见 `research-wiki/project_mechanism_goal_audit_20260907.md`。已有 taxonomy、文献机制地图、旧跨架构/匹配统计和当前 YOLO 阶段诊断应保留；主要缺口是把多个根因变成有独特预测与反证条件的并行假设，以及恢复旧跨架构产物并统一明确评测口径。O 大类可作为强一对多结构保留，不能把细分类局限当作大类逻辑错误。

状态更正：R005c 最新 `experiments/yolo26_confidence_audit_20260907_v2/audit_results.json` 已生成，`deterministic_verdict=fail`，与同文件 `acceptance_status=accepted` 冲突；stage/source 核对及 7 对置信度集合差异未澄清。此处不判定原数值错误，但暂停其“已审计通过”的使用。审查建议不是新方法或跨架构训练的自动授权；研究仍处于机制探索阶段。

## 执行授权记录

2026-09-07 审查改善已落实：当前证据状态及竞争假设以 `research-wiki/mechanism_evidence_ledger_20260907.md` 为入口。历史方案的实施批准已标为过时；R005c 官方 run Gate 记为 BLOCKED（对应原审计 FAIL），不得从上文历史“审计中”恢复。跨架构证据恢复和混杂重采样失败排查已列明完成条件，尚未重算。Faro 仍为当前优先诊断方向。

- `2026-09-07`：用户确认当前 `yolo26_task05_final.pt` 仅在 PigLife 上微调，训练归档的 manifest/image_root 与此一致；授权对 FaroPigSeg 全部 train/val/test（1,518 图）与 BamaPig2D 全部 train/eval（3,340 图）执行固定 checkpoint 推理。两者的原始 train 划分也是本模型的外部推理数据，不用于本轮训练。协议为 `experiments/yolo26_external_full_20260907_protocol.md`；既有 test/eval 的探索使用记录继续保留。

- `2026-09-04`：用户已明确授权文献检索员立即执行上述“下一阶段”的文献搜索、交叉核验、Research Wiki 入库与 gap map 更新。
- 本授权只覆盖文献与证据构建；不覆盖方法路线选择、网络或损失设计、训练、`research-refine`、`experiment-plan` 或 `research_contract`。
- 文献检索员应在共享工作区自主完成此范围内的工作，并将产物写入项目；不得就此范围再次要求用户确认。
- `2026-09-05`：用户追加授权主控统筹、Terra 执行本方向研究；可在 Conda `pytorch` 环境自主准备免费公开资源并执行诊断。以上 `2026-09-04` 条目仅是当时文献检索员的授权范围。当前继续既定诊断主线；重大方法方向须结合证据与用户决策，不能以输出集合 oracle 自动通过方法 Gate。
- `2026-09-07`：用户明确选择优先继续 Faro 单 mask 质量缺口诊断，覆盖框定位、mask 表达与标注语义；该选择是诊断优先级，不是网络、损失或论文贡献的接受。
